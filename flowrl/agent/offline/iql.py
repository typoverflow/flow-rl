from functools import partial
from typing import Tuple

import distrax
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.offline.algo.iql import IQLConfig
from flowrl.functional.ema import ema_update
from flowrl.functional.loss import expectile_regression
from flowrl.module.actor import SquashedDeterministicActor, TanhMeanGaussianActor
from flowrl.module.critic import Critic, EnsembleCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("deterministic_actor", "max_action", "min_action"))
def jit_sample_action(
    actor: Model,
    obs: jnp.ndarray,
    deterministic_actor: bool,
    max_action: float,
    min_action: float,
) -> jnp.ndarray:
    """
    Sample action from the actor network.
    """
    if deterministic_actor:
        action = actor(obs, training=False)
    else:
        dist = actor(obs, training=False)
        dist: distrax.Normal
        action = dist.mean()
        action = jnp.clip(action, min_action, max_action)
    return action

def update_v(
    value: Model,
    critic_target: Model,
    batch: Batch,
    expectile: float,
) -> Tuple[Model, Metric]:
    qs = critic_target(batch.obs, batch.action)
    qs = qs.min(axis=0)
    def value_loss_fn(value_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        v = value.apply(
            {"params": value_params},
            batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        value_loss = expectile_regression(v, qs, expectile).mean()
        return value_loss, {
            "loss/value_loss": value_loss,
            "misc/v_mean": v.mean(),
        }
    new_value, metrics = value.apply_gradient(value_loss_fn)
    return new_value, metrics

def update_q(
    critic: Model,
    value: Model,
    batch: Batch,
    discount: float
) -> Tuple[Model, Metric]:
    target_q = batch.reward + discount * (1-batch.terminal) * value(batch.next_obs)
    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        qs = critic.apply(
            {"params": critic_params},
            batch.obs,
            batch.action,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        critic_loss = ((qs - target_q[jnp.newaxis, ...])**2).sum(0).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": qs.mean()
        }
    new_critic, metrics = critic.apply_gradient(critic_loss_fn)
    return new_critic, metrics

def update_actor(
    actor: Model,
    critic_target: Model,
    value: Model,
    batch: Batch,
    beta: float,
    deterministic_actor: bool,
) -> Tuple[Model, Metric]:
    v = value(batch.obs)
    qs = critic_target(batch.obs, batch.action)
    qs = qs.min(axis=0)
    adv = qs - v
    exp_a = jnp.exp(adv * beta)
    exp_a = jnp.minimum(exp_a, 100.0)  # truncate the weights...
    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        pred = actor.apply(
            {"params": actor_params},
            batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        if deterministic_actor:
            actor_loss = jnp.sum((pred - batch.action) ** 2, axis=-1, keepdims=True)
        else:
            dist: distrax.MultivariateNormalDiag = pred
            actor_loss = - dist.log_prob(batch.action)[..., jnp.newaxis]
        actor_loss = (exp_a * actor_loss).mean()
        return actor_loss, {
            "loss/actor_loss": actor_loss,
            "misc/adv_mean": adv.mean(),
            "misc/weight_mean": exp_a.mean()
        }
    new_actor, metrics = actor.apply_gradient(actor_loss_fn)
    return new_actor, metrics

@partial(jax.jit, static_argnames=("expectile", "beta", "discount", "ema", "deterministic_actor"))
def update_iql(
    actor: Model,
    critic: Model,
    critic_target: Model,
    value: Model,
    batch: Batch,
    expectile: float,
    beta: float,
    discount: float,
    ema: float,
    deterministic_actor: bool,
)-> Tuple[Model, Model, Model, Model, Metric]:
    new_value, value_metrics = update_v(value, critic_target, batch, expectile)
    new_actor, actor_metrics = update_actor(actor, critic_target, new_value, batch, beta, deterministic_actor)
    new_critic, critic_metrics = update_q(critic, new_value, batch, discount)
    new_critic_target = ema_update(new_critic, critic_target, ema)
    return new_actor, new_critic, new_critic_target, new_value, {
        **value_metrics,
        **actor_metrics,
        **critic_metrics,
    }


class IQLAgent(BaseAgent):
    """
    Implicit Q-Learning (IQL)
    https://arxiv.org/abs/2110.06169
    """
    name = "IQLAgent"
    model_names = ["actor", "critic", "critic_target", "value"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: IQLConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_key, critic_key, value_key = jax.random.split(self.rng, 4)

        if cfg.deterministic_actor:
            actor_def = SquashedDeterministicActor(
                backbone=MLP(
                    hidden_dims=cfg.actor_hidden_dims,
                    layer_norm=False,
                    dropout=cfg.dropout
                ),
                obs_dim=self.obs_dim,
                action_dim=self.act_dim
            )
        else:
            actor_def = TanhMeanGaussianActor(
                backbone=MLP(
                    hidden_dims=cfg.actor_hidden_dims,
                    layer_norm=False,
                    dropout=cfg.dropout
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
            clip_grad_norm=cfg.clip_grad_norm,
        )

        critic_def = EnsembleCritic(
            hidden_dims=cfg.critic_hidden_dims,
            layer_norm=False,
            dropout=cfg.dropout,
            ensemble_size=cfg.critic_ensemble_size
        )
        self.critic = Model.create(
            critic_def,
            critic_key,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.critic_target = Model.create(
            critic_def,
            critic_key,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
        )

        value_def = Critic(
            hidden_dims=cfg.value_hidden_dims,
            layer_norm=cfg.layer_norm,
            dropout=cfg.dropout,
        )
        self.value = Model.create(
            value_def,
            value_key,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=optax.adam(learning_rate=cfg.value_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.actor, self.critic, self.critic_target, self.value, metric = update_iql(
            self.actor,
            self.critic,
            self.critic_target,
            self.value,
            batch,
            self.cfg.expectile,
            self.cfg.beta,
            self.cfg.discount,
            self.cfg.ema,
            self.cfg.deterministic_actor,
        )
        return metric

    def sample_actions(self, obs, deterministic=True, num_samples = 1):
        assert num_samples == 1, "IQL only supports num_samples=1"
        assert deterministic, "IQL only supports deterministic=True"
        action = jit_sample_action(
            self.actor,
            obs,
            self.cfg.deterministic_actor,
            max_action=self.cfg.max_action,
            min_action=self.cfg.min_action,
        )
        return action, {}
