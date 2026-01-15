from functools import partial
from typing import Tuple

import distrax
import flax
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.agent.online.brc.network import (
    BroNet,
    EnsembleRffBroNetCritic,
    FactorizedDDPM,
    update_factorized_ddpm,
)
from flowrl.config.online.algo.brc.diffsr_brcld import DiffSRBRCLDConfig
from flowrl.flow.langevin_dynamics import IBCLangevinDynamics
from flowrl.functional.ema import ema_update
from flowrl.module.misc import TunableCoefficient
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("training", "ld_temp", "num_samples", "num_bins", "v_max"))
def jit_sample_actions_ld(
    rng: PRNGKey,
    ld: Model,
    critic_target: Model,
    ddpm_target: Model,
    scaler: jnp.ndarray,
    obs: jnp.ndarray,
    training: bool,
    ld_temp: float,
    num_samples: int,
    num_bins: int,
    v_max: float,
):
    assert len(obs.shape) == 2
    B = obs.shape[0]
    rng, xT_rng = jax.random.split(rng)

    # sample
    obs_repeat = obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
    xT = jax.random.normal(xT_rng, (*obs_repeat.shape[:-1], ld.x_dim))
    bin_values = jnp.linspace(start=-v_max, stop=v_max, num=num_bins)

    def model_fn(xt, input_t, condition):
        original_shape = xt.shape[:-1]
        xt = xt.reshape(-1, xt.shape[-1])
        input_t = input_t.reshape(-1, 1)
        condition = condition.reshape(-1, condition.shape[-1])
        energy_and_grad_fn = jax.vmap(jax.value_and_grad(
            lambda xt, t, condition:
                (jax.nn.softmax(critic_target(ddpm_target(condition, xt, method="forward_phi")), axis=-1).mean(axis=0) \
                * bin_values).sum(-1)
        ))
        energy, grad = energy_and_grad_fn(xt, input_t, condition)
        energy = energy.reshape(*original_shape, 1)
        grad = grad.reshape(*original_shape, -1)
        grad = grad / ld_temp / (scaler + 1e-8)
        return energy, grad
    rng, actions, history = ld.sample(
        rng, model_fn, xT, obs_repeat, training,
    )
    if num_samples == 1:
        actions = actions[:, 0]
    else:
        feature = ddpm_target(obs_repeat, actions, method="forward_phi")
        q_logits = critic_target(feature)
        q_probs = jax.nn.softmax(q_logits, axis=-1).mean(axis=0)
        qs = (bin_values[jnp.newaxis, jnp.newaxis, ...] * q_probs).sum(-1, keepdims=True)
        qs = qs.reshape(B, num_samples)
        best_idx = qs.argmax(axis=-1)
        actions = actions.reshape(B, num_samples, -1)[jnp.arange(B), best_idx]
    return rng, actions, history

def update_q(
    rng: PRNGKey,
    ld: Model,
    actor: Model,
    log_alpha: Model,
    critic: Model,
    critic_target: Model,
    ddpm_target: Model,
    scaler: jnp.ndarray,
    batch: Batch,
    discount: float,
    ld_temp: float,
    num_bins: int,
    v_max: float,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, sample_rng = jax.random.split(rng)
    rng, _, history = jit_sample_actions_ld(
        rng,
        ld,
        critic_target,
        ddpm_target,
        scaler,
        batch.next_obs,
        training=False,
        ld_temp=ld_temp,
        num_samples=1,
        num_bins=num_bins,
        v_max=v_max,
    )
    dist = actor(batch.next_obs)
    next_action, next_logprob = dist.sample_and_log_prob(seed=sample_rng)
    next_feature = ddpm_target(batch.next_obs, next_action, method="forward_phi")
    next_q_logits = critic_target(next_feature)
    next_q_probs = jax.nn.softmax(next_q_logits, axis=-1).mean(axis=0)
    v_min = - v_max
    bin_values = jnp.linspace(start=v_min, stop=v_max, num=num_bins)[jnp.newaxis]
    delta = (v_max - v_min) / (num_bins - 1)
    target_bin_values = batch.reward + discount * (1 - batch.terminal) * (
        bin_values
        - jnp.exp(log_alpha()) * next_logprob
    )
    target_bin_values = jnp.clip(target_bin_values, v_min, v_max)
    target_bin_values = (target_bin_values - v_min) / delta
    lower, upper = jnp.floor(target_bin_values), jnp.ceil(target_bin_values)
    lower_mask = jax.nn.one_hot(lower.reshape(-1), num_bins).reshape(-1, num_bins, num_bins)
    upper_mask = jax.nn.one_hot(upper.reshape(-1), num_bins).reshape(-1, num_bins, num_bins)
    lower_values = (next_q_probs * (upper + (lower == upper).astype(jnp.float32) - target_bin_values))[..., None]
    upper_values = (next_q_probs * (target_bin_values - lower))[..., None]
    target_probs = jax.lax.stop_gradient(jnp.sum(lower_values * lower_mask + upper_values * upper_mask, axis=1))
    q_value_target = (bin_values * target_probs).sum(-1)

    feature = ddpm_target(batch.obs, batch.action, method="forward_phi")
    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        q_logits = critic.apply(
            {"params": critic_params},
            feature,
            training=False,
            rngs={"dropout": dropout_rng},
        )
        q_logprobs = jax.nn.log_softmax(q_logits, axis=-1)
        critic_loss = -(target_probs[jnp.newaxis] * q_logprobs).sum(-1).mean(-1).sum(0)
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": q_value_target.mean(),
            "misc/q_min": q_value_target.min(),
            "misc/q_max": q_value_target.max(),
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

def update_actor(
    rng: PRNGKey,
    actor: Model,
    log_alpha: Model,
    critic_target: Model,
    ddpm_target: Model,
    batch: Batch,
    num_bins: int,
    v_max: float,
):
    rng, sample_rng = jax.random.split(rng)
    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey):
        dist = actor.apply(
            {"params": actor_params},
            batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        new_action, new_logprob = dist.sample_and_log_prob(seed=sample_rng)
        new_feature = ddpm_target(batch.obs, new_action, method="forward_phi")
        q_logits = critic_target(new_feature)
        q_probs = jax.nn.softmax(q_logits, axis=-1).mean(axis=0)
        bin_values = jnp.linspace(start=-v_max, stop=v_max, num=num_bins)[jnp.newaxis]
        q_values = (bin_values * q_probs).sum(-1, keepdims=True)
        actor_loss = (jnp.exp(log_alpha()) * new_logprob - q_values).mean()
        return actor_loss, {
            "loss/actor_loss": actor_loss,
            "misc/entropy": -new_logprob.mean(),
        }
    new_actor, metrics = actor.apply_gradient(actor_loss_fn)
    return rng, new_actor, metrics

def update_alpha(
    rng: PRNGKey,
    log_alpha: Model,
    actor: Model,
    batch: Batch,
    target_entropy: float,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, sample_rng = jax.random.split(rng)
    dist = actor(batch.obs)
    action, logprob = dist.sample_and_log_prob(seed=sample_rng)

    def alpha_loss_fn(log_alpha_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        log_alpha_value = log_alpha.apply(
            {"params": log_alpha_params},
        )
        loss = - log_alpha_value * (logprob + target_entropy).mean()
        return loss, {
            "loss/alpha_loss": loss,
            "misc/alpha": jnp.exp(log_alpha_value),
        }
    new_log_alpha, metrics = log_alpha.apply_gradient(alpha_loss_fn)
    return rng, new_log_alpha, metrics

def update_diffsr_brcld(
    rng: PRNGKey,
    ld: Model,
    actor: Model,
    log_alpha: Model,
    critic: Model,
    critic_target: Model,
    ddpm: Model,
    ddpm_target: Model,
    scaler: jnp.ndarray,
    batch: Batch,
    discount: float,
    ema: float,
    ld_temp: float,
    num_bins: int,
    v_max: float,
    reward_coef: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Metric]:
    rng, new_ddpm, ddpm_metrics = update_factorized_ddpm(rng, ddpm, batch, reward_coef)
    rng, new_critic, new_scaler, critic_metrics = update_q(rng, ld, actor, log_alpha, critic, critic_target, ddpm_target, scaler, batch, discount, ld_temp, num_bins, v_max)
    rng, new_actor, actor_metrics = update_actor(rng, actor, log_alpha, critic_target, ddpm_target, batch, num_bins, v_max)
    rng, new_log_alpha, alpha_metrics = update_alpha(rng, log_alpha, new_actor, batch, -9.5)
    new_critic_target = ema_update(new_critic, critic_target, ema)
    new_ddpm_target = ema_update(ddpm, ddpm_target, ema)
    return rng, new_actor, new_log_alpha, new_critic, new_critic_target, new_ddpm, new_ddpm_target, new_scaler, {
        **critic_metrics,
        **ddpm_metrics,
        **actor_metrics,
        **alpha_metrics,
    }

@partial(jax.jit, static_argnames=("num_updates", "discount", "ema", "ld_temp", "num_bins", "v_max", "reward_coef"))
def multiple_update_diffsr_brcld(
    num_updates: int,
    rng: PRNGKey,
    ld: Model,
    actor: Model,
    log_alpha: Model,
    critic: Model,
    critic_target: Model,
    ddpm: Model,
    ddpm_target: Model,
    scaler: jnp.ndarray,
    batch: Batch,
    discount: float,
    ema: float,
    ld_temp: float,
    num_bins: int,
    v_max: float,
    reward_coef: float,
):
    mini_batch_size = batch.obs.shape[0] // num_updates
    batch = jax.tree.map(lambda x: x.reshape((num_updates, mini_batch_size, -1)) if x is not None else None, batch)
    def one_update(i, state):
        rng, actor, log_alpha, critic, critic_target, ddpm, ddpm_target, scaler, metrics = state
        new_rng, new_actor, new_log_alpha, new_critic, new_critic_target, new_ddpm, new_ddpm_target, new_scaler, new_metrics = update_diffsr_brcld(
            rng,
            ld,
            actor,
            log_alpha,
            critic,
            critic_target,
            ddpm,
            ddpm_target,
            scaler,
            jax.tree.map(lambda x: jnp.take(x, i, axis=0) if x is not None else None, batch),
            discount,
            ema,
            ld_temp,
            num_bins,
            v_max,
            reward_coef,
        )
        return new_rng, new_actor, new_log_alpha, new_critic, new_critic_target, new_ddpm, new_ddpm_target, new_scaler, new_metrics

    rng, actor, log_alpha, critic, critic_target, ddpm, ddpm_target, scaler, metrics = one_update(0, (rng, actor, log_alpha, critic, critic_target, ddpm, ddpm_target, scaler, {}))
    return jax.lax.fori_loop(1, num_updates, one_update, (rng, actor, log_alpha, critic, critic_target, ddpm, ddpm_target, scaler, metrics))


class DiffSRBRCLDAgent(BaseAgent):
    """
    Bigger, Regularized, Categorical (BRC) agent.
    """
    name = "DiffSRBRCLDAgent"
    model_names = ["ddpm", "ddpm_target", "actor", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: DiffSRBRCLDConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg

        self.reward_coef = cfg.reward_coef
        self.num_noises = cfg.num_noises
        self.feature_dim = cfg.feature_dim

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

        critic_def = EnsembleRffBroNetCritic(
            hidden_dim=cfg.critic_hidden_dim,
            num_blocks=1,
            output_dim=cfg.num_bins,
            ensemble_size=cfg.critic_ensemble_size,
        )

        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.feature_dim)), ),
            optimizer=optax.adamw(learning_rate=cfg.critic_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.feature_dim)), ),
        )

        self.ld = IBCLangevinDynamics.create(
            network=flax.linen.Dense(1),
            rng=ld_rng,
            inputs=(jnp.ones((1, 1))),
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

        # this is just a placeholder
        self.log_alpha = Model.create(
            TunableCoefficient(init_value=0.0),
            jax.random.PRNGKey(0),
            inputs=(),
            optimizer=optax.adam(learning_rate=3e-4, b1=0.5),
        )

        # the backup actor
        from flowrl.module.actor import SquashedGaussianActor
        actor_def = SquashedGaussianActor(
            backbone=BroNet(
                hidden_dim=256,
                num_blocks=1,
            ),
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            conditional_logstd=True,
            logstd_softclip=True,
        )
        self.actor = Model.create(
            actor_def,
            ld_rng,
            inputs=(jnp.ones((1, self.obs_dim))),
            optimizer=optax.adamw(learning_rate=3e-4),
        )
        self._n_training_steps = 0


    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.actor, self.log_alpha, self.critic, self.critic_target, self.ddpm, self.ddpm_target, self.scaler, metrics = multiple_update_diffsr_brcld(
            self.cfg.num_updates,
            self.rng,
            self.ld,
            self.actor,
            self.log_alpha,
            self.critic,
            self.critic_target,
            self.ddpm,
            self.ddpm_target,
            self.scaler,
            batch,
            discount=self.cfg.discount,
            ema=self.cfg.ema,
            ld_temp=self.cfg.ld_temp,
            num_bins=self.cfg.num_bins,
            v_max=self.cfg.v_max,
            reward_coef=self.reward_coef,
        )
        self._n_training_steps += 1
        return metrics

    def sample_actions(self, obs, deterministic=True, num_samples=1):
        if deterministic:
            num_samples = self.cfg.num_samples
            self.rng, action, history = jit_sample_actions_ld(
                self.rng,
                self.ld,
                self.critic_target,
                self.ddpm_target,
                self.scaler,
                obs,
                training=False,
                ld_temp=self.cfg.ld_temp,
                num_samples=num_samples,
                num_bins=self.cfg.num_bins,
                v_max=self.cfg.v_max,
            )
        else:
            num_samples = 1
            from flowrl.agent.online.brc.diffsr_brc import jit_sample_action
            self.rng, sample_key = jax.random.split(self.rng)
            action = jit_sample_action(
                sample_key,
                self.actor,
                obs,
                deterministic,
            )
        # if not deterministic:
        #     action = action + self.cfg.exploration_noise * jax.random.normal(self.rng, action.shape)
        #     action = jnp.clip(action, -1.0, 1.0)
        return action, {}
