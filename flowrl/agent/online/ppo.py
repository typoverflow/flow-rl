from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.online.mujoco.algo.ppo import PPOConfig
from flowrl.module.actor import GaussianActor
from flowrl.module.critic import Critic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.types import *
from flowrl.types import Metric, Param, PRNGKey


class PPOBatch(NamedTuple):
    """Batch for PPO training."""
    obs: jnp.ndarray
    action: jnp.ndarray
    logprob: jnp.ndarray
    advantage: jnp.ndarray
    return_to_go: jnp.ndarray
    value: jnp.ndarray


@partial(jax.jit, static_argnames=("clip_coef", "entropy_coef", "value_coef", "clip_vloss"))
def update_ppo(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    batch: PPOBatch,
    clip_coef: float,
    entropy_coef: float,
    value_coef: float,
    clip_vloss: bool,
) -> Tuple[PRNGKey, Model, Model, Metric]:
    """
    Update PPO actor and critic networks using clipped policy optimization.
    """

    def loss_fn(params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        # Actor forward pass
        dist = actor.apply(
            {"params": params["actor"]},
            batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )

        # Critic forward pass
        value = critic.apply(
            {"params": params["critic"]},
            batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )

        # Get new log probabilities
        new_logprob = dist.log_prob(batch.action)

        # Calculate ratio for clipped policy objective
        logratio = new_logprob - batch.logprob
        ratio = jnp.exp(logratio)

        # Policy loss with clipping
        pg_loss1 = -batch.advantage * ratio
        pg_loss2 = -batch.advantage * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss with optional clipping
        value_pred = value.squeeze(-1)
        if clip_vloss:
            v_loss_unclipped = (value_pred - batch.return_to_go) ** 2
            v_clipped = batch.value + jnp.clip(
                value_pred - batch.value,
                -clip_coef,
                clip_coef,
            )
            v_loss_clipped = (v_clipped - batch.return_to_go) ** 2
            v_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((value_pred - batch.return_to_go) ** 2).mean()

        # Entropy bonus
        entropy = dist.entropy().mean()
        entropy_loss = -entropy

        # Total loss
        total_loss = pg_loss + value_coef * v_loss + entropy_coef * entropy_loss

        # Approximate KL divergence for early stopping
        approx_kl = ((ratio - 1) - logratio).mean()

        # Clip fraction
        clipfrac = ((ratio - 1.0).abs() > clip_coef).mean()

        metrics = {
            "loss/total_loss": total_loss,
            "loss/policy_loss": pg_loss,
            "loss/value_loss": v_loss,
            "loss/entropy": entropy,
            "misc/approx_kl": approx_kl,
            "misc/clipfrac": clipfrac,
            "misc/explained_variance": 1 - jnp.var(batch.return_to_go - value_pred) / (jnp.var(batch.return_to_go) + 1e-8),
            "misc/ratio": ratio.mean(),
        }

        return total_loss, metrics

    # Combine parameters
    params = {"actor": actor.params, "critic": critic.params}

    # Compute gradients and update
    rng, dropout_rng = jax.random.split(rng)
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, dropout_rng)

    # Update actor
    new_actor = actor.apply_gradient(lambda p: grads["actor"])
    new_critic = critic.apply_gradient(lambda p: grads["critic"])

    return rng, new_actor, new_critic, metrics


@partial(jax.jit, static_argnames=("deterministic",))
def jit_sample_action(
    rng: PRNGKey,
    actor: Model,
    obs: jnp.ndarray,
    deterministic: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample actions from the actor network.
    """
    dist = actor(obs, training=False)
    if deterministic:
        action = dist.mode()
    else:
        action = dist.sample(seed=rng)
    logprob = dist.log_prob(action)
    return action, logprob


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent.
    """
    name = "PPOAgent"
    model_names = ["actor", "critic"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_rng, critic_rng = jax.random.split(self.rng, 3)

        # Actor network
        actor_def = GaussianActor(
            backbone=MLP(
                hidden_dims=cfg.actor_hidden_dims,
                layer_norm=cfg.layer_norm,
                dropout=None,
            ),
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            conditional_logstd=False,
        )

        # Critic network
        critic_def = Critic(
            hidden_dims=cfg.critic_hidden_dims,
            layer_norm=cfg.layer_norm,
            dropout=None,
        )

        self.actor = Model.create(
            actor_def,
            actor_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=optax.adamw(learning_rate=cfg.lr),
            clip_grad_norm=cfg.max_grad_norm,
        )

        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=optax.adamw(learning_rate=cfg.lr),
            clip_grad_norm=cfg.max_grad_norm,
        )

    def train_step(self, batch: PPOBatch, step: int) -> Metric:
        """
        Perform a PPO training step.
        Note: This expects the batch to contain pre-computed advantages and returns.
        """
        self.rng, self.actor, self.critic, metrics = update_ppo(
            self.rng,
            self.actor,
            self.critic,
            batch,
            clip_coef=self.cfg.clip_coef,
            entropy_coef=self.cfg.entropy_coef,
            value_coef=self.cfg.value_coef,
            clip_vloss=self.cfg.clip_vloss,
        )
        return metrics

    def sample_actions(
        self,
        obs: jnp.ndarray,
        deterministic: bool = False,
        num_samples: int = 1,
    ) -> Tuple[jnp.ndarray, Metric]:
        """
        Sample actions from the policy.
        """
        assert num_samples == 1, "PPO only supports num_samples=1"
        self.rng, sample_key = jax.random.split(self.rng)
        action, logprob = jit_sample_action(
            sample_key,
            self.actor,
            obs,
            deterministic,
        )
        return action, {"logprob": logprob}
