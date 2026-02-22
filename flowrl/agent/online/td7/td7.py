from functools import partial
from typing import Any, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.agent.online.td7.network import TD7Actor, TD7Encoder, TD7EnsembleCritic
from flowrl.config.online.algo.td7 import TD7Config
from flowrl.functional.ema import ema_update
from flowrl.functional.misc import sg
from flowrl.module.model import Model
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("deterministic", "exploration_noise"))
def jit_sample_action(
    rng: PRNGKey,
    actor: Model,
    encoder: Model,
    obs: jnp.ndarray,
    deterministic: bool,
    exploration_noise: float,
) -> jnp.ndarray:
    zs = encoder(obs, method="zs")
    action = actor(obs, zs)
    if not deterministic:
        action = action + exploration_noise * jax.random.normal(rng, action.shape)
        action = jnp.clip(action, -1.0, 1.0)
    return action


@partial(jax.jit, static_argnames=("discount", "policy_noise", "noise_clip", "max_action"))
def update_critic(
    rng: PRNGKey,
    critic: Model,
    critic_target: Model,
    actor_target: Model,
    fixed_encoder: Model,
    fixed_encoder_target: Model,
    batch: Batch,
    discount: float,
    policy_noise: float,
    noise_clip: float,
    max_action: float,
    min_target_q: float,
    max_target_q: float,
) -> Tuple[PRNGKey, Model, Metric, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    rng, noise_rng = jax.random.split(rng)

    # Get next state embeddings
    fixed_target_next_zs = fixed_encoder_target(batch.next_obs, method="zs")

    # Add noise to target actions
    noise = jax.random.normal(noise_rng, batch.action.shape) * policy_noise
    noise = jnp.clip(noise, -noise_clip, noise_clip)
    next_action = actor_target(batch.next_obs, fixed_target_next_zs)
    next_action = jnp.clip(next_action + noise, -max_action, max_action)

    # Get next state-action embeddings
    fixed_target_next_zsa = fixed_encoder_target(fixed_target_next_zs, next_action, method="zsa")

    # Compute target Q values
    q_target = critic_target(
        batch.next_obs,
        next_action,
        fixed_target_next_zsa,
        fixed_target_next_zs,
    ).min(axis=0)

    q_target = batch.reward + discount * (1 - batch.terminal) * jnp.clip(q_target, min_target_q, max_target_q)

    # Get current embeddings
    fixed_zs = fixed_encoder(batch.obs, method="zs")
    fixed_zsa = fixed_encoder(fixed_zs, batch.action, method="zsa")

    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        q = critic.apply(
            {"params": critic_params},
            batch.obs,
            batch.action,
            fixed_zsa,
            fixed_zs,
            training=True,
            rngs={"dropout": dropout_rng},
        )

        td = jnp.abs(q_target[jnp.newaxis, :] - q)
        critic_loss = jnp.where(td < 1.0, 0.5 * td**2, td).sum(axis=0).mean()

        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_pred": q.mean(),
            "misc/max_q": q_target.max(),
            "misc/min_q": q_target.min(),
            "priority": jnp.max(td, axis=0),
        }

    new_critic, metrics = critic.apply_gradient(critic_loss_fn)

    priority = metrics.pop("priority")
    return rng, new_critic, metrics, q_target.max(), q_target.min(), priority.squeeze()


@partial(jax.jit, static_argnames=("lam"))
def update_actor(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    fixed_encoder: Model,
    batch: Batch,
    lam: float,
) -> Tuple[PRNGKey, Model, Metric]:
    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        fixed_zs = fixed_encoder(batch.obs, method="zs")
        new_action = actor.apply(
            {"params": actor_params},
            batch.obs,
            fixed_zs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        fixed_zsa = fixed_encoder(fixed_zs, new_action, method="zsa")
        q = critic(batch.obs, new_action, fixed_zsa, fixed_zs, training=True, rngs={"dropout": dropout_rng})

        actor_q_loss = -q.mean()

        if lam > 0:
            bc_loss = sg(jnp.abs(q).mean()) * jnp.mean((new_action - batch.action)**2)
        else:
            bc_loss = 0.0

        actor_loss = actor_q_loss + lam * bc_loss

        return actor_loss, {
            "loss/actor_q_loss": actor_q_loss,
            "loss/bc_loss": bc_loss,
        }

    new_actor, metrics = actor.apply_gradient(actor_loss_fn)
    return rng, new_actor, metrics


@jax.jit
def update_encoder(
    rng: PRNGKey,
    encoder: Model,
    fixed_encoder_target: Model,
    batch: Batch,
) -> Tuple[PRNGKey, Model, Metric]:
    next_zs = encoder(batch.next_obs, method="zs")
    def encoder_loss_fn(encoder_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        zs_pred = encoder.apply(
            {"params": encoder_params}, batch.obs, method="zs", training=True, rngs={"dropout": dropout_rng}
        )
        zsa_pred = encoder.apply(
            {"params": encoder_params}, zs_pred, batch.action, method="zsa", training=True, rngs={"dropout": dropout_rng}
        )

        encoder_loss = jnp.mean((zsa_pred - next_zs)**2)

        return encoder_loss, {
            "loss/encoder_loss": encoder_loss,
        }

    new_encoder, metrics = encoder.apply_gradient(encoder_loss_fn)
    return rng, new_encoder, metrics


class TD7Agent(BaseAgent):
    """
    TD7 Agent: State-Action Representation Learning for Deep Reinforcement Learning
    https://arxiv.org/abs/2306.02451
    """
    name = "TD7Agent"
    model_names = ["actor", "actor_target", "critic", "critic_target", "encoder", "fixed_encoder", "fixed_encoder_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: TD7Config, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)

        self.cfg = cfg
        self.lam = cfg.lam
        self.actor_update_freq = cfg.actor_update_freq
        self.target_update_freq = cfg.target_update_freq
        self.exploration_noise = cfg.exploration_noise

        self.rng, actor_rng, critic_rng, encoder_rng = jax.random.split(self.rng, 4)

        self._max_target_q = 0.0
        self._min_target_q = 0.0
        self._max_target_q_uptodate = -1e8
        self._min_target_q_uptodate = 1e8

        # Create encoder
        encoder_def = TD7Encoder(
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            activation=nn.elu,
        )

        # Create actor
        actor_def = TD7Actor(
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            activation=nn.relu,
        )

        # Create critic
        critic_def = TD7EnsembleCritic(
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            ensemble_size=2,
            activation=nn.elu,
        )

        # Initialize models
        self.encoder = Model.create(
            encoder_def,
            encoder_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
            optimizer=optax.adam(learning_rate=cfg.encoder_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )

        self.fixed_encoder = Model.create(
            encoder_def,
            encoder_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
        )

        self.fixed_encoder_target = Model.create(
            encoder_def,
            encoder_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
        )

        self.actor = Model.create(
            actor_def,
            actor_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, cfg.embed_dim))),
            optimizer=optax.adam(learning_rate=cfg.actor_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )

        self.actor_target = Model.create(
            actor_def,
            actor_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, cfg.embed_dim))),
        )

        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(
                jnp.ones((1, self.obs_dim)),
                jnp.ones((1, self.act_dim)),
                jnp.ones((1, cfg.embed_dim)),
                jnp.ones((1, cfg.embed_dim)),
            ),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )

        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(
                jnp.ones((1, self.obs_dim)),
                jnp.ones((1, self.act_dim)),
                jnp.ones((1, cfg.embed_dim)),
                jnp.ones((1, cfg.embed_dim)),
            ),
        )

        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        metrics = {}
        cfg = self.cfg

        # Update encoder
        self.rng, self.encoder, encoder_metrics = update_encoder(
            self.rng, self.encoder, self.fixed_encoder_target, batch
        )
        metrics.update(encoder_metrics)

        # Update critic
        self.rng, self.critic, critic_metrics, max_q, min_q, priority = update_critic(
            self.rng,
            self.critic,
            self.critic_target,
            self.actor_target,
            self.fixed_encoder,
            self.fixed_encoder_target,
            batch,
            discount=cfg.discount,
            policy_noise=cfg.target_policy_noise,
            noise_clip=cfg.noise_clip,
            max_action=cfg.max_action,
            min_target_q=self._min_target_q,
            max_target_q=self._max_target_q,
        )
        metrics.update(critic_metrics)

        # Update target Q bounds
        self._max_target_q_uptodate = max(self._max_target_q_uptodate, max_q.item())
        self._min_target_q_uptodate = min(self._min_target_q_uptodate, min_q.item())

        # Update actor
        if self._n_training_steps % self.actor_update_freq == 0:
            self.rng, self.actor, actor_metrics = update_actor(
                self.rng,
                self.actor,
                self.critic,
                self.fixed_encoder,
                batch,
                lam=self.lam,
            )
            metrics.update(actor_metrics)

        # Update targets
        if self._n_training_steps % self.target_update_freq == 0:
            self.actor_target = ema_update(self.actor, self.actor_target, 1.0)
            self.critic_target = ema_update(self.critic, self.critic_target, 1.0)
            self.fixed_encoder_target = ema_update(self.fixed_encoder, self.fixed_encoder_target, 1.0)
            self.fixed_encoder = ema_update(self.encoder, self.fixed_encoder, 1.0)
            self._max_target_q = self._max_target_q_uptodate
            self._min_target_q = self._min_target_q_uptodate

        self._n_training_steps += 1
        return {
            **metrics,
            "priority": priority,
        }

    def sample_actions(self, obs, deterministic=True, num_samples=1):
        assert num_samples == 1, "TD7 only supports num_samples=1"
        self.rng, sample_rng = jax.random.split(self.rng)
        action = jit_sample_action(
            sample_rng,
            self.actor,
            self.fixed_encoder,
            obs,
            deterministic,
            exploration_noise=self.cfg.exploration_noise,
        )
        return action, {}
