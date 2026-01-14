from functools import partial
from typing import Tuple

import distrax
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.offline.algo.ivr import IVRConfig
from flowrl.functional.ema import ema_update
from flowrl.module.actor import TanhMeanGaussianActor
from flowrl.module.critic import Critic, EnsembleCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("max_action", "min_action"))
def jit_sample_action(
    actor: Model,
    obs: jnp.ndarray,
    max_action: float,
    min_action: float,
) -> jnp.ndarray:
    dist = actor(obs, training=False)
    dist: distrax.Normal
    action = dist.mean()
    action = jnp.clip(action, min_action, max_action)
    return action

@partial(jax.jit, static_argnames=("discount"))
def update_q(
    critic: Model,
    value: Model,
    batch: Batch,
    discount: float
) -> Tuple[Model, Metric]:
    next_v = value(batch.next_obs)
    target_q = batch.reward + discount * (1-batch.terminal) * next_v
    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        qs = critic.apply(
            {"params": critic_params},
            batch.obs,
            batch.action,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        critic_loss = ((qs - target_q[jnp.newaxis, ...])**2).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": qs.mean()
        }
    new_critic, metrics = critic.apply_gradient(critic_loss_fn)
    return new_critic, metrics

@partial(jax.jit, static_argnames=("alpha", "method"))
def update_v(
    value: Model,
    critic_target: Model,
    batch: Batch,
    alpha: float,
    method: str
) -> Tuple[Model, Metric]:
    qs = critic_target(batch.obs, batch.action)
    qs = qs.min(axis=0)

    def value_loss_fn(value_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        v = value.apply(
            {"params": value_params},
            batch.obs,
            training=True,
            rngs={"dropout": dropout_rng}
        )
        if method == "sql":
            sp_term = (qs-v) / (2*alpha) + 1.0
            sp_weight = jnp.where(sp_term>0, 1., 0.)
            value_loss = (sp_weight * (sp_term**2) + v / alpha).mean()
        elif method == "eql":
            sp_term = (qs - v) / alpha
            sp_term = jnp.minimum(sp_term, 5.0)
            max_sp_term = jnp.max(sp_term, axis=0)
            max_sp_term = jnp.where(max_sp_term < -1.0, -1.0, max_sp_term)
            max_sp_term = jax.lax.stop_gradient(max_sp_term)
            value_loss = (jnp.exp(sp_term - max_sp_term) + jnp.exp(-max_sp_term) * v / alpha).mean()
        else:
            raise NotImplementedError(f"Unsupported method: {method}")
        return value_loss, {
            "loss/value_loss": value_loss,
            "misc/v": v.mean(),
            "misc/q-v": (qs-v).mean()
        }
    new_value, metrics = value.apply_gradient(value_loss_fn)
    return new_value, metrics

@partial(jax.jit, static_argnames=("alpha", "method"))
def update_actor(
    actor: Model,
    critic_target: Model,
    value: Model,
    batch: Batch,
    alpha: float,
    method: str
) -> Tuple[Model, Metric]:
    v = value(batch.obs)
    qs = critic_target(batch.obs, batch.action)
    qs = qs.min(axis=0)
    if method == "sql":
        weight = qs - v
    elif method == "eql":
        weight = jnp.exp(10*(qs-v)/alpha)
    else:
        raise NotImplementedError(f"Unsupported method: {method}")
    weight = jnp.clip(weight, 0., 100.)

    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        dist = actor.apply(
            {"params": actor_params},
            batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        logprobs = dist.log_prob(batch.action).sum(-1, keepdims=True)
        actor_loss = -(weight * logprobs).mean()
        return actor_loss, {
            "loss/actor_loss": actor_loss,
        }
    new_actor, metrics = actor.apply_gradient(actor_loss_fn)
    return new_actor, metrics

@partial(jax.jit, static_argnames=("discount", "ema", "alpha", "method"))
def update_ivr(
    actor: Model,
    critic: Model,
    critic_target: Model,
    value: Model,
    batch: Batch,
    discount: float,
    ema: float,
    alpha: float,
    method: str
) -> Tuple[Model, Model, Model, Model, Metric]:
    new_value, value_metrics = update_v(value, critic_target, batch, alpha, method)
    new_actor, actor_metrics = update_actor(actor, critic_target, new_value, batch, alpha, method)
    new_critic, critic_metrics = update_q(critic, new_value, batch, discount)
    new_critic_target = ema_update(new_critic, critic_target, ema)
    return new_actor, new_critic, new_critic_target, new_value, {
        **value_metrics,
        **actor_metrics,
        **critic_metrics
    }

class IVRAgent(BaseAgent):
    """
    Implicit Value Regularization (IVR) agent.
    """
    name = "IVRAgent"
    model_names = ["actor", "critic", "critic_target", "value"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: IVRConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_key, critic_key, value_key = jax.random.split(self.rng, 4)

        actor_def = TanhMeanGaussianActor(
            backbone=MLP(
                hidden_dims=cfg.actor_hidden_dims,
                layer_norm=False,
                dropout=cfg.actor_dropout
            ),
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            conditional_logstd=cfg.conditional_logstd,
            logstd_min=cfg.policy_logstd_min
        )
        if cfg.opt_decay_schedule == "cosine" and cfg.lr_decay_steps is not None:
            lr = optax.cosine_decay_schedule(cfg.actor_lr, cfg.lr_decay_steps)
            act_opt = optax.adam(learning_rate=lr)
        else:
            act_opt = optax.adam(learning_rate=cfg.actor_lr)
        self.actor = Model.create(
            actor_def,
            actor_key,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=act_opt,
        )

        critic_def = EnsembleCritic(
            hidden_dims=cfg.critic_hidden_dims,
            layer_norm=False,
            ensemble_size=2,
        )
        self.critic = Model.create(
            critic_def,
            critic_key,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
        )
        self.critic_target = Model.create(
            critic_def,
            critic_key,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
        )

        value_def = Critic(
            hidden_dims=cfg.value_hidden_dims,
            layer_norm=cfg.layer_norm,
            dropout=cfg.value_dropout,
        )
        self.value = Model.create(
            value_def,
            value_key,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=optax.adam(learning_rate=cfg.value_lr),
        )

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.actor, self.critic, self.critic_target, self.value, metrics = update_ivr(
            self.actor,
            self.critic,
            self.critic_target,
            self.value,
            batch,
            self.cfg.discount,
            self.cfg.ema,
            self.cfg.alpha,
            self.cfg.method
        )
        return metrics

    def sample_actions(self, obs, deterministic=True, num_samples=1):
        assert num_samples == 1, "IVR only supports num_samples=1"
        assert deterministic, "IVR only supports deterministic=True"
        action = jit_sample_action(
            self.actor,
            obs,
            max_action=self.cfg.max_action,
            min_action=self.cfg.min_action
        )
        return action, {}
