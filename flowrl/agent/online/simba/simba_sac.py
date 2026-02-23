from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.agent.online.simba.network import SimbaCritic, SimbaNet
from flowrl.config.online.algo.simba.simba_sac import SimbaSACConfig
from flowrl.functional.ema import ema_update
from flowrl.module.actor import SquashedGaussianActor
from flowrl.module.misc import PositiveTunableCoefficient
from flowrl.module.model import Model
from flowrl.types import *
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

def update_critic(
    rng: PRNGKey,
    critic: Model,
    critic_target: Model,
    actor: Model,
    alpha: Model,
    batch: Batch,
    discount: float,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, sample_rng = jax.random.split(rng)
    dist = actor(batch.next_obs)
    next_action, next_logprob = dist.sample_and_log_prob(seed=sample_rng)
    next_q = critic_target(batch.next_obs, next_action)
    target_q = batch.reward + discount * (1-batch.terminal) * (next_q.min(axis=0) - alpha() * next_logprob[..., jnp.newaxis])

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
    alpha: Model,
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
        actor_loss = (alpha() * new_logprob - q.mean(axis=0)).mean()
        return actor_loss, {
            "loss/actor_loss": actor_loss,
        }
    new_actor, metrics = actor.apply_gradient(actor_loss_fn)
    return rng, new_actor, metrics

def update_alpha(
    rng: PRNGKey,
    alpha: Model,
    actor: Model,
    batch: Batch,
    target_entropy: float,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, sample_rng = jax.random.split(rng)
    dist = actor(batch.obs)
    action, logprob = dist.sample_and_log_prob(seed=sample_rng)

    def alpha_loss_fn(alpha_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        alpha_value = alpha.apply(
            {"params": alpha_params},
        )
        loss = - alpha_value * (logprob + target_entropy).mean()
        return loss, {
            "loss/alpha_loss": loss,
            "misc/alpha": alpha_value,
            "misc/entropy": -logprob.mean(),

        }
    new_alpha, metrics = alpha.apply_gradient(alpha_loss_fn)
    return rng, new_alpha, metrics

def update_sac_once(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    critic_target: Model,
    alpha: Model,
    batch: Batch,
    discount: float,
    ema: float,
    target_entropy: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Metric]:
    rng, new_critic, critic_metrics = update_critic(rng, critic, critic_target, actor, alpha, batch, discount)
    rng, new_actor, actor_metrics = update_actor(rng, actor, new_critic, alpha, batch)
    rng, new_alpha, alpha_metrics = update_alpha(rng, alpha, new_actor, batch, target_entropy)
    new_critic_target = ema_update(new_critic, critic_target, ema)
    return rng, new_actor, new_critic, new_critic_target, new_alpha, {
        **critic_metrics,
        **actor_metrics,
        **alpha_metrics,
    }

@partial(jax.jit, static_argnames=("num_updates", "discount", "ema", "target_entropy"))
def update_sac_multiple(
    num_updates: int,
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    critic_target: Model,
    alpha: Model,
    batch: Batch,
    discount: float,
    ema: float,
    target_entropy: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Metric]:
    mini_batch_size = batch.obs.shape[0] // num_updates
    batch = jax.tree.map(lambda x: x.reshape((num_updates, mini_batch_size, -1)) if x is not None else None, batch)
    def one_update(i, state):
        rng, actor, critic, critic_target, alpha, metrics = state
        new_rng, new_actor, new_critic, new_critic_target, new_alpha, new_metrics = update_sac_once(
            rng, actor, critic, critic_target, alpha, jax.tree.map(lambda x: jnp.take(x, i, axis=0) if x is not None else None, batch), discount, ema, target_entropy)
        return new_rng, new_actor, new_critic, new_critic_target, new_alpha, new_metrics
    rng, actor, critic, critic_target, alpha, metrics = one_update(0, (rng, actor, critic, critic_target, alpha, {}))
    return jax.lax.fori_loop(1, num_updates, one_update, (rng, actor, critic, critic_target, alpha, metrics))


class SimbaSACAgent(BaseAgent):
    """
    Simba Soft Actor-Critic (SAC) agent.
    """
    name = "SimbaSACAgent"
    model_names = ["actor", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: SimbaSACConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_rng, critic_rng, alpha_rng = jax.random.split(self.rng, 4)

        self.num_updates = cfg.num_updates

        actor_def = SquashedGaussianActor(
            backbone=SimbaNet(
                num_blocks=cfg.actor_num_blocks,
                hidden_dim=cfg.actor_hidden_dim,
            ),
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            conditional_logstd=True,
            logstd_softclip=True,
        )
        critic_def = nn.vmap(
            SimbaCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=cfg.critic_ensemble_size,
        )(
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
        )

        self.actor = Model.create(
            actor_def,
            actor_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=optax.adamw(learning_rate=cfg.actor_lr, weight_decay=cfg.actor_wd),
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
            optimizer=optax.adamw(learning_rate=cfg.critic_lr, weight_decay=cfg.critic_wd),
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
        )
        self.alpha = Model.create(
            PositiveTunableCoefficient(init_value=cfg.alpha_init_value),
            alpha_rng,
            inputs=(),
            optimizer=optax.adamw(learning_rate=cfg.alpha_lr, weight_decay=cfg.alpha_wd),
        )
        self.target_entropy = cfg.target_entropy_scale * self.act_dim

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.actor, self.critic, self.critic_target, self.alpha, metrics = update_sac_multiple(
            num_updates=self.num_updates,
            rng=self.rng,
            actor=self.actor,
            critic=self.critic,
            critic_target=self.critic_target,
            alpha=self.alpha,
            batch=batch,
            discount=self.cfg.discount,
            ema=self.cfg.ema,
            target_entropy=self.target_entropy,
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
