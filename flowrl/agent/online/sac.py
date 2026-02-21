from functools import partial
from typing import Tuple

import distrax
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.online.algo.sac import SACConfig
from flowrl.functional.ema import ema_update
from flowrl.module.actor import SquashedGaussianActor
from flowrl.module.critic import Ensemblize, ScalarCritic
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
) -> Tuple[PRNGKey, Model, Metric]:
    rng, sample_rng = jax.random.split(rng)
    dist = actor(batch.next_obs)
    next_action, next_logprob = dist.sample_and_log_prob(seed=sample_rng)
    next_q = critic_target(batch.next_obs, next_action)
    target_q = batch.reward + discount * (1-batch.terminal) * (next_q.min(axis=0) - jnp.exp(log_alpha()) * next_logprob)

    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        q = critic.apply(
            {"params": critic_params},
            batch.obs,
            batch.action,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        critic_loss = ((q - target_q[jnp.newaxis, :])**2).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": q.mean(),
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
        q = critic(batch.obs, new_action)
        actor_loss = (jnp.exp(log_alpha()) * new_logprob - q.min(axis=0)).mean()
        return actor_loss, {
            "loss/actor_loss": actor_loss,
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

@partial(jax.jit, static_argnames=("discount", "ema", "target_entropy"))
def update_sac(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    critic_target: Model,
    log_alpha: Model,
    batch: Batch,
    discount: float,
    ema: float,
    target_entropy: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Metric]:
    rng, new_critic, critic_metrics = update_q(rng, critic, critic_target, actor, log_alpha, batch, discount)
    rng, new_actor, actor_metrics = update_actor(rng, actor, new_critic, log_alpha, batch)
    rng, new_log_alpha, alpha_metrics = update_alpha(rng, log_alpha, new_actor, batch, target_entropy)
    new_critic_target = ema_update(new_critic, critic_target, ema)
    return rng, new_actor, new_critic, new_critic_target, new_log_alpha, {
        **critic_metrics,
        **actor_metrics,
        **alpha_metrics,
    }

class SACAgent(BaseAgent):
    """
    Soft Actor-Critic (SAC) agent.
    """
    name = "SACAgent"
    model_names = ["actor", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: SACConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_rng, critic_rng, alpha_rng = jax.random.split(self.rng, 4)

        activation = {
            "relu": jax.nn.relu,
            "elu": jax.nn.elu,
        }[cfg.activation]
        actor_def = SquashedGaussianActor(
            backbone=MLP(
                hidden_dims=cfg.actor_hidden_dims,
                layer_norm=cfg.layer_norm,
                activation=activation,
                dropout=None,
            ),
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            conditional_logstd=True,
        )
        critic_def = Ensemblize(
            base=ScalarCritic(
                backbone=MLP(
                    hidden_dims=cfg.critic_hidden_dims,
                    layer_norm=cfg.layer_norm,
                    activation=activation,
                    dropout=None,
                ),
            ),
            ensemble_size=cfg.critic_ensemble_size,
        )
        self.actor = Model.create(
            actor_def,
            actor_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=optax.adam(learning_rate=cfg.actor_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
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
            optimizer=optax.adam(learning_rate=cfg.alpha_lr),
        )
        self.target_entropy = -self.act_dim

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.actor, self.critic, self.critic_target, self.log_alpha, metrics = update_sac(
            self.rng,
            self.actor,
            self.critic,
            self.critic_target,
            self.log_alpha,
            batch,
            self.cfg.discount,
            self.cfg.ema,
            self.target_entropy,
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
