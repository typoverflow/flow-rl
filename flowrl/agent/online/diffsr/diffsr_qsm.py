from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.online.diffsr.network import FactorizedDDPM, update_factorized_ddpm
from flowrl.agent.online.qsm import QSMAgent
from flowrl.config.online.mujoco.algo.diffsr import DiffSRQSMConfig
from flowrl.flow.continuous_ddpm import ContinuousDDPM
from flowrl.functional.ema import ema_update
from flowrl.module.actor import SquashedDeterministicActor
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.rff import RffEnsembleCritic
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("training", "num_samples", "solver"))
def jit_sample_actions(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    ddpm_target: Model,
    obs: jnp.ndarray,
    training: bool,
    num_samples: int,
    solver: str,
) -> Tuple[PRNGKey, jnp.ndarray]:
    assert len(obs.shape) == 2
    B = obs.shape[0]
    rng, xT_rng = jax.random.split(rng)

    # sample
    obs_repeat = obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
    xT = jax.random.normal(xT_rng, (*obs_repeat.shape[:-1], actor.x_dim))
    rng, actions, _ = actor.sample(rng, xT, obs_repeat, training, solver)
    if num_samples == 1:
        actions = actions[:, 0]
    else:
        feature = ddpm_target(obs_repeat, actions, method="forward_phi")
        qs = critic(feature)
        qs = qs.min(axis=0).reshape(B, num_samples)
        best_idx = qs.argmax(axis=-1)
        actions = actions.reshape(B, num_samples, -1)[jnp.arange(B), best_idx]
    return rng, actions

@partial(jax.jit, static_argnames=("discount", "solver"))
def update_critic(
    rng: PRNGKey,
    critic: Model,
    critic_target: Model,
    actor: ContinuousDDPM,
    ddpm_target: Model,
    batch: Batch,
    discount: float,
    solver: str,
    critic_coef: float
) -> Tuple[PRNGKey, Model, Metric]:
    rng, sample_rng = jax.random.split(rng)
    next_xT = jax.random.normal(sample_rng, (*batch.next_obs.shape[:-1], actor.x_dim))
    rng, next_action, _ = actor.sample(
        rng,
        next_xT,
        batch.next_obs,
        training=False,
        solver=solver,
    )
    next_feature = ddpm_target(batch.next_obs, next_action, method="forward_phi")
    q_target = critic_target(next_feature).min(0)
    q_target = batch.reward + discount * (1 - batch.terminal) * q_target

    feature = ddpm_target(batch.obs, batch.action, method="forward_phi")

    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        q_pred = critic.apply(
            {"params": critic_params},
            feature,
            rngs={"dropout": dropout_rng},
        )
        critic_loss = critic_coef * ((q_pred - q_target[jnp.newaxis, :])**2).sum(0).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": q_pred.mean(),
            "misc/reward": batch.reward.mean(),
        }

    new_critic, metrics = critic.apply_gradient(critic_loss_fn)
    return rng, new_critic, metrics


@partial(jax.jit, static_argnames=("temp"))
def update_actor(
    rng: PRNGKey,
    actor: Model,
    ddpm_target: ContinuousDDPM,
    critic_target: Model,
    batch: Batch,
    temp: float,
) -> Tuple[PRNGKey, Model, Metric]:

    a0 = batch.action
    rng, at, t, eps = actor.add_noise(rng, a0)
    alpha1, alpha2 = actor.noise_schedule_func(t)

    def get_q_value(action: jnp.ndarray, obs: jnp.ndarray) -> jnp.ndarray:
        feature = ddpm_target(obs, action, method="forward_phi")
        q = critic_target(feature)
        return q.min(axis=0).mean()
    q_grad_fn = jax.vmap(jax.grad(get_q_value))
    q_grad = q_grad_fn(at, batch.obs)
    eps_estimation = - alpha2 * q_grad / temp / (jnp.abs(q_grad).mean() + 1e-6)

    def loss_fn(diffusion_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        eps_pred = actor.apply(
            {"params": diffusion_params},
            at,
            t,
            condition=batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        loss = ((eps_pred - eps_estimation) ** 2).mean()
        return loss, {
            "loss/actor_loss": loss,
            "misc/eps_estimation_l1": jnp.abs(eps_estimation).mean(),
            "misc/eps_estimation_std": jnp.std(eps_estimation, axis=0).mean(),
        }

    new_actor, actor_metrics = actor.apply_gradient(loss_fn)
    return rng, new_actor, actor_metrics

# @jax.jit
# def jit_compute_metrics(
#     rng: PRNGKey,
#     critic: Model,
#     ddpm_target: Model,
#     diffusion_value: Model,
#     diffusion_actor: Model,
#     batch: Batch,
# ) -> Tuple[PRNGKey, Metric]:
#     B, S = batch.obs.shape
#     A = batch.action.shape[-1]
#     num_actions = 50
#     metrics = {}
#     rng, action_rng = jax.random.split(rng)
#     obs_repeat = batch.obs[..., jnp.newaxis, :].repeat(num_actions, axis=-2)
#     action_repeat = jax.random.uniform(action_rng, (B, num_actions, A), minval=-1.0, maxval=1.0)

#     def get_critic(at, obs):
#         t1 = jnp.ones((1, ), dtype=jnp.int32)
#         ft = ddpm_target(obs, at, t1, method="forward_phi")
#         q = critic(ft)
#         return q.mean()
#     all_critic, all_critic_grad = jax.vmap(jax.value_and_grad(get_critic))(
#         action_repeat.reshape(-1, A),
#         obs_repeat.reshape(-1, S),
#     )
#     all_critic = all_critic.reshape(B, num_actions, 1)
#     all_critic_grad = all_critic_grad.reshape(B, num_actions, -1)
#     metrics.update({
#         f"q_std/critic": all_critic.std(axis=1).mean(),
#         f"q_grad/critic": jnp.abs(all_critic_grad).mean(),
#     })

#     def get_value(at, obs, t):
#         ft = ddpm_target(obs, at, t, method="forward_phi")
#         q = diffusion_value(ft)
#         return q.mean()
#     for t in [0] + list(range(1, diffusion_actor.steps+1, diffusion_actor.steps//5)):
#         t_input = jnp.ones((B, num_actions, 1)) * t
#         all_value, all_value_grad = jax.vmap(jax.value_and_grad(get_value))(
#             action_repeat.reshape(-1, A),
#             obs_repeat.reshape(-1, S),
#             t_input.reshape(-1, 1),
#         )
#         all_value = all_value.reshape(B, num_actions, 1)
#         all_value_grad = all_value_grad.reshape(B, num_actions, -1)
#         metrics.update({
#             f"q_std/value_{t}": all_value.std(axis=1).mean(),
#             f"q_grad/value_{t}": jnp.abs(all_value_grad).mean(),
#         })
#     return rng, metrics

class DiffSRQSMAgent(QSMAgent):
    """
    Diff-SR with QSM agent.
    """

    name = "DiffSRQSMAgent"
    model_names = ["ddpm", "ddpm_target", "actor", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: DiffSRQSMConfig, seed: int):
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
        self.temp = cfg.temp

        # networks
        self.rng, ddpm_rng, ddpm_init_rng, actor_rng, critic_rng = jax.random.split(self.rng, 5)
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
            optimizer=optax.adam(learning_rate=cfg.feature_lr),
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
            ensemble_size=2,
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.feature_dim)),),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.feature_dim)),),
        )

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

        self.rng, self.critic, critic_metrics = update_critic(
            self.rng,
            self.critic,
            self.critic_target,
            self.actor,
            self.ddpm_target,
            batch,
            discount=self.cfg.discount,
            solver=self.cfg.diffusion.solver,
            critic_coef=self.critic_coef,
        )
        metrics.update(critic_metrics)

        if self._n_training_steps % self.actor_update_freq == 0:
            self.rng, self.actor, actor_metrics = update_actor(
                self.rng,
                self.actor,
                self.ddpm_target,
                self.critic_target,
                batch,
                temp=self.temp,
            )
            metrics.update(actor_metrics)

        if self._n_training_steps % self.target_update_freq == 0:
            self.sync_target()

        # if self._n_training_steps % 2000 == 0:
        #     self.rng, metrics = jit_compute_metrics(
        #         self.rng,
        #         self.critic,
        #         self.ddpm_target,
        #         self.diffusion_value,
        #         self.diffusion_actor,
        #         batch,
        #     )
        #     metrics.update(metrics)
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
        self.rng, action = jit_sample_actions(
            self.rng,
            self.actor,
            self.critic,
            self.ddpm_target,
            obs,
            training=False,
            num_samples=num_samples,
            solver=self.cfg.diffusion.solver,
        )
        if not deterministic:
            action = action + 0.1 * jax.random.normal(self.rng, action.shape)
        return action, {}

    def sync_target(self):
        self.critic_target = ema_update(self.critic, self.critic_target, self.cfg.ema)
        self.ddpm_target = ema_update(self.ddpm, self.ddpm_target, self.cfg.feature_ema)
