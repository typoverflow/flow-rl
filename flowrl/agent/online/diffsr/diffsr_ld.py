import os
from functools import partial
from typing import Tuple

import flax
import jax
import jax.numpy as jnp
import optax

import flowrl.module.initialization as init
from flowrl.agent.base import BaseAgent
from flowrl.agent.online.diffsr.network import FactorizedDDPM, update_factorized_ddpm
from flowrl.config.online.algo.diffsr import DiffSRLDConfig
from flowrl.flow.langevin_dynamics import IBCLangevinDynamics
from flowrl.functional.ema import ema_update
from flowrl.module.model import Model
from flowrl.module.rff import RffEnsembleCritic
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("training", "ld_temp", "num_samples"))
def jit_sample_actions_ld(
    rng: PRNGKey,
    ld: Model,
    ddpm_target: Model,
    critic_target: Model,
    scaler: jnp.ndarray,
    obs: jnp.ndarray,
    training: bool,
    ld_temp: float,
    num_samples: int,
) -> Tuple[PRNGKey, jnp.ndarray, jnp.ndarray]:
    assert len(obs.shape) == 2
    B = obs.shape[0]
    rng, xT_rng = jax.random.split(rng)

    # sample
    obs_repeat = obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
    xT = jax.random.normal(xT_rng, (*obs_repeat.shape[:-1], ld.x_dim))

    def model_fn(xt, input_t, condition):
        original_shape = xt.shape[:-1]
        xt = xt.reshape(-1, xt.shape[-1])
        input_t = input_t.reshape(-1, 1)
        condition = condition.reshape(-1, condition.shape[-1])
        energy_and_grad_fn = jax.vmap(jax.value_and_grad(
            lambda xt, t, condition: critic_target(ddpm_target(condition, xt, method="forward_phi")).mean()
        ))
        energy, grad = energy_and_grad_fn(xt, input_t, condition)
        energy = energy.reshape(*original_shape, 1)
        grad = grad.reshape(*original_shape, -1)
        grad = grad / ld_temp / (scaler + 1e-8)
        return energy, grad

    rng, actions, history = ld.sample(
        rng,
        model_fn,
        xT,
        obs_repeat,
    )
    if num_samples == 1:
        actions = actions[:, 0]
    else:
        feature = ddpm_target(obs_repeat, actions, method="forward_phi")
        qs = critic_target(feature)
        qs = qs.mean(axis=0).reshape(B, num_samples)
        best_idx = qs.argmax(axis=-1)
        actions = actions.reshape(B, num_samples, -1)[jnp.arange(B), best_idx]

    return rng, actions, history

@partial(jax.jit, static_argnames=("discount", "ld_temp"))
def update_critic(
    rng: PRNGKey,
    ld: Model,
    critic: Model,
    critic_target: Model,
    ddpm_target: Model,
    scaler: jnp.ndarray,
    batch: Batch,
    discount: float,
    ld_temp: float,
) -> Tuple[PRNGKey, Model, jnp.ndarray, Metric]:
    rng, sample_rng, q_rng = jax.random.split(rng, 3)
    rng, next_action, history = jit_sample_actions_ld(
        rng,
        ld,
        ddpm_target,
        critic_target,
        scaler,
        batch.next_obs,
        training=False,
        ld_temp=ld_temp,
        num_samples=1,
    )
    next_feature = ddpm_target(batch.next_obs, next_action, method="forward_phi")
    q_target = critic_target(next_feature)
    q_index = jax.random.choice(q_rng, q_target.shape[0], shape=(2,), replace=False)
    q_target = q_target[q_index].min(0)
    q_target = batch.reward + discount * (1 - batch.terminal) * q_target

    feature = ddpm_target(batch.obs, batch.action, method="forward_phi")

    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        q_pred = critic.apply(
            {"params": critic_params},
            feature,
            rngs={"dropout": dropout_rng},
        )
        critic_loss = ((q_pred - q_target[jnp.newaxis, :])**2).sum(0).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": q_pred.mean(),
            "misc/reward": batch.reward.mean(),
        }

    # update scaler
    q_grad = history[1] * ld_temp * scaler
    new_scaler = 0.995 * scaler + 0.005 * jnp.abs(q_grad).mean()

    new_critic, metrics = critic.apply_gradient(critic_loss_fn)
    metrics.update({
        "misc_ld/scaler": new_scaler.mean(),
        "misc_ld/q_grad_l1": jnp.abs(q_grad).mean(),
    })
    return rng, new_critic, new_scaler, metrics


class DiffSRLDAgent(BaseAgent):
    """
    Diff-SR with Langevin Dynamics Agent.
    """

    name = "DiffSRLDAgent"
    model_names = ["ddpm", "ddpm_target", "actor", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: DiffSRLDConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg

        self.ddpm_coef = cfg.ddpm_coef
        self.critic_coef = cfg.critic_coef
        self.reward_coef = cfg.reward_coef
        self.num_noises = cfg.num_noises
        self.feature_dim = cfg.feature_dim
        self.rff_dim = cfg.rff_dim
        self.actor_update_freq = cfg.actor_update_freq
        self.target_update_freq = cfg.target_update_freq

        # networks
        self.rng, ddpm_rng, ddpm_init_rng, ld_rng, critic_rng = jax.random.split(self.rng, 5)
        ddpm_def = FactorizedDDPM(
            self.obs_dim,
            self.act_dim,
            self.feature_dim,
            cfg.embed_dim,
            cfg.phi_hidden_dims,
            cfg.mu_hidden_dims,
            cfg.reward_hidden_dims,
            cfg.rff_dim,
            cfg.num_noises,
        )
        self.ddpm = Model.create(
            ddpm_def,
            ddpm_rng,
            inputs=(
                ddpm_init_rng,
                jnp.ones((1, self.obs_dim)),
                jnp.ones((1, self.act_dim)),
                jnp.ones((1, self.obs_dim)),
            ),
            optimizer=optax.adamw(learning_rate=cfg.feature_lr, weight_decay=cfg.wd),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.ddpm_target = Model.create(
            ddpm_def,
            ddpm_rng,
            inputs=(
                ddpm_init_rng,
                jnp.ones((1, self.obs_dim)),
                jnp.ones((1, self.act_dim)),
                jnp.ones((1, self.obs_dim)),
            ),
        )

        critic_def = RffEnsembleCritic(
            feature_dim=self.feature_dim,
            hidden_dims=cfg.critic_hidden_dims,
            rff_dim=cfg.rff_dim,
            ensemble_size=cfg.critic_ensemble_size,
            kernel_init=init.pytorch_kernel_init,
            bias_init=init.pytorch_bias_init,
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.feature_dim)),),
            optimizer=optax.adamw(learning_rate=cfg.critic_lr, weight_decay=cfg.wd),
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.feature_dim)),),
        )

        self.ld = IBCLangevinDynamics.create(
            rng=ld_rng,
            x_dim=self.act_dim,
            steps=self.cfg.ld.steps,
            schedule=self.cfg.ld.schedule,
            stepsize_init=self.cfg.ld.stepsize_init,
            stepsize_final=self.cfg.ld.stepsize_final,
            stepsize_decay=self.cfg.ld.stepsize_decay,
            stepsize_power=self.cfg.ld.stepsize_power,
            noise_scale=self.cfg.ld.noise_scale,
            grad_clip=self.cfg.ld.grad_clip,
            drift_clip=self.cfg.ld.drift_clip,
            margin_clip=self.cfg.ld.margin_clip,
        )
        self.scaler = jnp.ones((1, ), dtype=jnp.float32)

        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        metrics = {}
        self.rng, self.ddpm, ddpm_metrics = update_factorized_ddpm(
            self.rng,
            self.ddpm,
            batch,
            self.reward_coef,
        )
        metrics.update(ddpm_metrics)

        self.rng, self.critic, self.scaler, critic_metrics = update_critic(
            self.rng,
            self.ld,
            self.critic,
            self.critic_target,
            self.ddpm_target,
            self.scaler,
            batch,
            discount=self.cfg.discount,
            ld_temp=self.cfg.ld_temp,
        )
        metrics.update(critic_metrics)

        if self._n_training_steps % self.target_update_freq == 0:
            self.sync_target()

        self._n_training_steps += 1
        return metrics

    def sample_actions(
        self,
        obs: jnp.ndarray,
        deterministic: bool = True,
        num_samples: int = 1,
    ) -> Tuple[jnp.ndarray, Metric]:
        if deterministic:
            num_samples = self.cfg.num_samples
        else:
            num_samples = 1
        self.rng, action, history = jit_sample_actions_ld(
            self.rng,
            self.ld,
            self.ddpm_target,
            self.critic_target,
            self.scaler,
            obs,
            training=False,
            ld_temp=self.cfg.ld_temp,
            num_samples=num_samples,
        )
        if not deterministic:
            action = action + self.cfg.exploration_noise * jax.random.normal(self.rng, action.shape)
            action = jnp.clip(action, -1.0, 1.0)
        return action, {}

    def sync_target(self):
        self.critic_target = ema_update(self.critic, self.critic_target, self.cfg.ema)
        self.ddpm_target = ema_update(self.ddpm, self.ddpm_target, self.cfg.feature_ema)

    def save(self, path: str):
            super().save(path)
            jnp.save(os.path.join(os.getcwd(), path, "scaler.npy"), self.scaler)

    def load(self, path: str):
        super().load(path)
        self.scaler = jnp.load(os.path.join(os.getcwd(), path, "scaler.npy"))
