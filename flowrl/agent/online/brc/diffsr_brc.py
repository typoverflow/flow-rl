from functools import partial
from typing import Tuple

import distrax
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.agent.online.brc.brc import BRCAgent
from flowrl.agent.online.brc.network import (
    BroNet,
    EnsembleBroNetCritic,
    FactorizedDDPM,
    update_factorized_ddpm,
)
from flowrl.config.online.algo.brc.diffsr_brc import DiffSRBRCConfig
from flowrl.functional.ema import ema_update
from flowrl.module.actor import SquashedGaussianActor
from flowrl.module.critic import EnsembleCritic
from flowrl.module.misc import TunableCoefficient
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("deterministic"))
def jit_sample_action(
    rng: PRNGKey,
    actor: Model,
    obs: jnp.ndarray,
    deterministic: bool,
) -> jnp.ndarray:
    dist = actor(obs, training=False)
    if deterministic:
        action = dist.tanh_mean()
    else:
        action = dist.sample(seed=rng)
    return action

def update_q(
    rng: PRNGKey,
    critic: Model,
    critic_target: Model,
    actor: Model,
    log_alpha: Model,
    ddpm_target: Model,
    batch: Batch,
    discount: float,
    num_bins: int,
    v_max: float,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, sample_rng = jax.random.split(rng)
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
            training=True,
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
    new_critic, metrics = critic.apply_gradient(critic_loss_fn)
    return rng, new_critic, metrics

def update_actor(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    log_alpha: Model,
    ddpm_target: Model,
    batch: Batch,
    num_bins: int,
    v_max: float,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, sample_rng = jax.random.split(rng)
    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        dist = actor.apply(
            {"params": actor_params},
            batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        new_action, new_logprob = dist.sample_and_log_prob(seed=sample_rng)
        new_feature = ddpm_target(batch.obs, new_action, method="forward_phi")
        q_logits = critic(new_feature)
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

def update_diffsr_brc(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_alpha: Model,
    ddpm: Model,
    ddpm_target: Model,
    batch: Batch,
    discount: float,
    ema: float,
    target_entropy: float,
    num_bins: int,
    v_max: float,
    reward_coef: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Metric]:
    rng, new_ddpm, ddpm_metrics = update_factorized_ddpm(rng, ddpm, batch, reward_coef)
    rng, new_critic, critic_metrics = update_q(rng, critic, critic_target, actor, log_alpha, ddpm_target, batch, discount, num_bins, v_max)
    rng, new_actor, actor_metrics = update_actor(rng, actor, new_critic, log_alpha, ddpm_target, batch, num_bins, v_max)
    rng, new_log_alpha, alpha_metrics = update_alpha(rng, log_alpha, new_actor, batch, target_entropy)
    new_critic_target = ema_update(new_critic, critic_target, ema)
    new_ddpm_target = ema_update(ddpm, ddpm_target, ema)
    return rng, new_actor, new_critic, new_critic_target, new_log_alpha, new_ddpm, new_ddpm_target, {
        **critic_metrics,
        **actor_metrics,
        **alpha_metrics,
        **ddpm_metrics,
    }

@partial(jax.jit, static_argnames=("num_updates", "discount", "ema", "target_entropy", "num_bins", "v_max", "reward_coef"))
def multiple_update_diffsr_brc(
    num_updates: int,
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_alpha: Model,
    ddpm: Model,
    ddpm_target: Model,
    batch: Batch,
    discount: float,
    ema: float,
    target_entropy: float,
    num_bins: int,
    v_max: float,
    reward_coef: float,
):
    mini_batch_size = batch.obs.shape[0] // num_updates
    batch = jax.tree.map(lambda x: x.reshape((num_updates, mini_batch_size, -1)) if x is not None else None, batch)
    def one_update(i, state):
        rng, actor, critic, critic_target, log_alpha, ddpm, ddpm_target, metrics = state
        new_rng, new_actor, new_critic, new_critic_target, new_log_alpha, new_ddpm, new_ddpm_target, new_metrics = update_diffsr_brc(
            rng,
            actor,
            critic,
            critic_target,
            log_alpha,
            ddpm,
            ddpm_target,
            jax.tree.map(lambda x: jnp.take(x, i, axis=0) if x is not None else None, batch),
            discount,
            ema,
            target_entropy,
            num_bins,
            v_max,
            reward_coef,
        )
        return new_rng, new_actor, new_critic, new_critic_target, new_log_alpha, new_ddpm, new_ddpm_target, new_metrics

    rng, actor, critic, critic_target, log_alpha, ddpm, ddpm_target, metrics = one_update(0, (rng, actor, critic, critic_target, log_alpha, ddpm, ddpm_target, {}))
    return jax.lax.fori_loop(1, num_updates, one_update, (rng, actor, critic, critic_target, log_alpha, ddpm, ddpm_target, metrics))


class DiffSRBRCAgent(BRCAgent):
    """
    Bigger, Regularized, Categorical (BRC) agent.
    """
    name = "DiffSRBRCAgent"
    model_names = ["ddpm", "ddpm_target", "actor", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: DiffSRBRCConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg

        self.reward_coef = cfg.reward_coef
        self.num_noises = cfg.num_noises
        self.feature_dim = cfg.feature_dim

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

        critic_def = EnsembleBroNetCritic(
            hidden_dim=cfg.critic_hidden_dim,
            num_blocks=2,
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

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.actor, self.critic, self.critic_target, self.log_alpha, self.ddpm, self.ddpm_target, metrics = multiple_update_diffsr_brc(
            self.cfg.num_updates,
            self.rng,
            self.actor,
            self.critic,
            self.critic_target,
            self.log_alpha,
            self.ddpm,
            self.ddpm_target,
            batch,
            self.cfg.discount,
            self.cfg.ema,
            self.target_entropy,
            self.num_bins,
            self.v_max,
            self.reward_coef,
        )
        return metrics

    def sample_actions(self, obs, deterministic=True, num_samples=1):
        assert num_samples == 1, "SAC only supports num_samples=1"
        self.rng, sample_key = jax.random.split(self.rng)
        action = jit_sample_action(
            sample_key,
            self.actor,
            obs,
            deterministic,
        )
        return action, {}
