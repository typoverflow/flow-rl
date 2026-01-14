from functools import partial
from typing import Tuple

import distrax
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.agent.online.brc.network import BroNet, EnsembleBroNetCritic
from flowrl.config.online.algo.brc.brc import BRCConfig
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
    batch: Batch,
    discount: float,
    num_bins: int,
    v_max: float,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, sample_rng = jax.random.split(rng)
    dist = actor(batch.next_obs)
    next_action, next_logprob = dist.sample_and_log_prob(seed=sample_rng)
    next_q_logits = critic_target(batch.next_obs, next_action)
    next_q_probs = jax.nn.softmax(next_q_logits, axis=-1).mean(axis=0)
    v_min = - v_max
    bin_values = jnp.linspace(start=v_min, stop=v_max, num=num_bins)[jnp.newaxis]
    delta = (v_max - v_min) / (num_bins - 1)
    target_bin_values = batch.reward + discount * (1 - batch.terminal) * (
        bin_values - jnp.exp(log_alpha()) * next_logprob
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

    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        q_logits = critic.apply(
            {"params": critic_params},
            batch.obs,
            batch.action,
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
        q_logits = critic(batch.obs, new_action)
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

@partial(jax.jit, static_argnames=("num_updates", "discount", "ema", "target_entropy", "num_bins", "v_max"))
def update_brc(
    num_updates: int,
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_alpha: Model,
    batch: Batch,
    discount: float,
    ema: float,
    target_entropy: float,
    num_bins: int,
    v_max: float,
):
    mini_batch_size = batch.obs.shape[0] // num_updates
    batch = jax.tree.map(lambda x: x.reshape((num_updates, mini_batch_size, -1)) if x is not None else None, batch)

    def one_update(i, state):
        rng, actor, critic, critic_target, log_alpha, _ = state
        mini_batch = jax.tree.map(lambda x: jnp.take(x, i, axis=0) if x is not None else None, batch)
        rng, new_critic, critic_metrics = update_q(rng, critic, critic_target, actor, log_alpha, mini_batch, discount, num_bins, v_max)
        rng, new_actor, actor_metrics = update_actor(rng, actor, new_critic, log_alpha, mini_batch, num_bins, v_max)
        rng, new_log_alpha, alpha_metrics = update_alpha(rng, log_alpha, new_actor, mini_batch, target_entropy)
        new_critic_target = ema_update(new_critic, critic_target, ema)
        metrics = {**critic_metrics, **actor_metrics, **alpha_metrics}
        return rng, new_actor, new_critic, new_critic_target, new_log_alpha, metrics

    state = one_update(0, (rng, actor, critic, critic_target, log_alpha, {}))
    return jax.lax.fori_loop(1, num_updates, one_update, state)


class BRCAgent(BaseAgent):
    """
    Bigger, Regularized, Categorical (BRC) agent.
    """
    name = "BRCAgent"
    model_names = ["actor", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: BRCConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_rng, critic_rng, alpha_rng = jax.random.split(self.rng, 4)

        self.num_bins = cfg.num_bins
        self.v_max = cfg.v_max

        actor_def = SquashedGaussianActor(
            backbone=BroNet(
                hidden_dim=cfg.actor_hidden_dim,
                num_blocks=1,
            ),
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            conditional_logstd=True,
            logstd_softclip=True,
        )
        critic_def = EnsembleBroNetCritic(
            hidden_dim=cfg.critic_hidden_dim,
            num_blocks=2,
            output_dim=cfg.num_bins,
            ensemble_size=cfg.critic_ensemble_size,
        )

        self.actor = Model.create(
            actor_def,
            actor_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=optax.adamw(learning_rate=cfg.actor_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
            optimizer=optax.adamw(learning_rate=cfg.critic_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
        )
        self.log_alpha = Model.create(
            TunableCoefficient(init_value=0.0),
            alpha_rng,
            inputs=(),
            optimizer=optax.adam(learning_rate=cfg.alpha_lr, b1=0.5),
        )
        self.target_entropy = -self.act_dim / 2

    def train_step(self, batch: Batch, step: int, num_updates: int = 1) -> Metric:
        self.rng, self.actor, self.critic, self.critic_target, self.log_alpha, metrics = update_brc(
            num_updates,
            self.rng,
            self.actor,
            self.critic,
            self.critic_target,
            self.log_alpha,
            batch,
            self.cfg.discount,
            self.cfg.ema,
            self.target_entropy,
            self.num_bins,
            self.v_max,
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
