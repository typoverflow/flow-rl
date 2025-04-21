from functools import partial
from typing import Tuple

import distrax
import jax
import flax.linen as nn
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.d4rl.algo.fql import FQLConfig
from flowrl.functional.ema import ema_update
from flowrl.flow.flow import OneStepTransform, VelocityField, FlowMatching, jit_update_flow_matching
from flowrl.module.critic import Critic, EnsembleCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.types import Batch, Metric, Param, PRNGKey

@partial(jax.jit, static_argnames=("max_action", "min_action", "act_dim"))
def jit_sample_action(
    key: PRNGKey,
    onestep_actor: Model,
    obs: jnp.ndarray,
    act_dim: int,
    max_action: float,
    min_action: float,
):
    B, _ = obs.shape
    x0 = jax.random.normal(key, (B, act_dim))
    action = onestep_actor(obs, x0)
    action = jnp.clip(action, min_action, max_action)
    return action

def update_critic(
    critic: Model,
    next_q: jnp.ndarray,
    batch: Batch,
    discount: float,
) -> Tuple[Model, Metric]:
    target_q = batch.reward + discount * (1-batch.terminal) * next_q
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

def update_onestep_actor(
    key: PRNGKey,
    onestep_actor: Model,
    actor_bc_flow: FlowMatching,
    critic: Model,
    obs: jnp.ndarray,
    alpha: float,
    normalize_q_loss: bool,
    act_dim: int,
    max_action: float,
    min_action: float,
) -> Tuple[Model, Metric]:
    B, _ = obs.shape
    noise = jax.random.normal(key, (B, act_dim))
    flow_actions, _ = actor_bc_flow.sample(
        key,
        obs,
        training=False,
        sample_noise=False,
        noise=noise,
    )
    def onestep_loss_fn(onestep_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        pred = onestep_actor.apply(
            {"params": onestep_params},
            obs,
            noise,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        distill_loss = ((pred - flow_actions) ** 2).mean()
        actions = jnp.clip(pred, min_action, max_action)
        qs = critic(obs, actions)
        q = jnp.mean(qs, axis=0) # always use mean aggregation here, follow the official implementation
        if normalize_q_loss:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = - lam * q.mean()
        else:
            q_loss = - q.mean()
        loss = alpha * distill_loss + q_loss
        return loss, {
            "loss/onestep_loss": loss,
            "loss/distill_loss": distill_loss,
            "loss/q_loss": q_loss,
            "misc/q_mean": q.mean(),
        }
    new_onestep_actor, metrics = onestep_actor.apply_gradient(onestep_loss_fn)
    return new_onestep_actor, metrics

@partial(jax.jit, static_argnames=("alpha", "normalize_q_loss", "discount", "tau", "q_agg", "max_action", "min_action"))
def jit_update_fql(
    rng: PRNGKey,
    actor_bc_flow: FlowMatching,
    actor_onestep: Model,
    critic: Model,
    critic_target: Model,
    batch: Batch,
    alpha: float,
    normalize_q_loss: bool,
    discount: float,
    tau: float,
    q_agg: str,
    max_action: float,
    min_action: float,
)-> Tuple[PRNGKey, Model, Model, Model, Model, Metric]:
    """
    Update the FQL agent.
    """
    rng, next_act_key, bc_key, onestep_key = jax.random.split(rng, 4)
    # update critic
    next_actions = jit_sample_action(
        next_act_key,
        actor_onestep,
        batch.next_obs,
        batch.action.shape[-1],
        max_action,
        min_action
    )
    next_qs = critic_target(batch.next_obs, next_actions)
    if q_agg == "mean":
        next_q = jnp.mean(next_qs, axis=0)
    elif q_agg == "min":
        next_q = jnp.min(next_qs, axis=0)
    else:
        raise ValueError(f"Unknown q_agg: {q_agg}")
    new_critic, critic_metrics = update_critic(
        critic,
        next_q,
        batch,
        discount
    )
    # update bc flow
    new_bc, bc_metrics = jit_update_flow_matching(
        bc_key,
        actor_bc_flow,
        batch,
    )
    # update onestep actor
    new_onestep_actor, onestep_metrics = update_onestep_actor(
        onestep_key,
        actor_onestep,
        actor_bc_flow,
        critic,
        batch.obs,
        alpha,
        normalize_q_loss,
        batch.action.shape[-1],
        max_action,
        min_action,
    )
    # update target critic
    new_critic_target = ema_update(new_critic, critic_target, tau)
    return rng, new_bc, new_onestep_actor, new_critic, new_critic_target, {
        **bc_metrics,
        **onestep_metrics,
        **critic_metrics,
    }


class FQLAgent(BaseAgent):
    """
    FQL (Flow Q-Learning, https://arxiv.org/abs/2502.02538) agent.
    """
    name = "FQLAgent"
    model_names = ["critic", "target_critic", "actor_bc_flow", "actor_onestep"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: FQLConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, bc_key, critic_key, onestep_key = jax.random.split(self.rng, 4)

        flow_def = VelocityField(
            backbone=MLP(
                hidden_dims=cfg.actor_hidden_dims,
                layer_norm=cfg.actor_layer_norm,
                output_dim=act_dim,
                activation=nn.gelu,
            ),
        )

        self.actor_bc_flow = FlowMatching.create(
            flow_def,
            bc_key,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim)), jnp.ones((1, 1))),
            steps=cfg.flow_steps,
            x_dim=act_dim,
            clip_sampler=False,
            optimizer=optax.adam(learning_rate=cfg.lr),
        )

        self.actor_onestep = Model.create(
            OneStepTransform(
                backbone=MLP(
                    hidden_dims=cfg.actor_hidden_dims,
                    layer_norm=cfg.actor_layer_norm,
                    output_dim=act_dim,
                    activation=nn.gelu,
                ),
            ),
            onestep_key,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
            optimizer=optax.adam(learning_rate=cfg.lr),
        )

        critic_def = EnsembleCritic(
            hidden_dims=cfg.critic_hidden_dims,
            layer_norm=cfg.critic_layer_norm,
            activation=nn.gelu,
            ensemble_size=2
        )
        self.critic = Model.create(
            critic_def,
            critic_key,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
            optimizer=optax.adam(learning_rate=cfg.lr),
        )
        self.critic_target = Model.create(
            critic_def,
            critic_key,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
        )

    def train_step(self, batch, step: int):
        self.rng, self.actor_bc_flow, self.actor_onestep, self.critic, self.critic_target, metrics = jit_update_fql(
            self.rng,
            self.actor_bc_flow,
            self.actor_onestep,
            self.critic,
            self.critic_target,
            batch,
            self.cfg.alpha,
            self.cfg.normalize_q_loss,
            self.cfg.discount,
            self.cfg.tau,
            self.cfg.q_agg,
            self.cfg.max_action,
            self.cfg.min_action
        )
        return metrics

    def sample_actions(self, obs, deterministic=True, num_samples = 1):
        assert num_samples==1, "FQL only supports num_samples=1"
        self.rng, key = jax.random.split(self.rng)
        action = jit_sample_action(
            key,
            self.actor_onestep,
            obs,
            self.act_dim,
            self.cfg.max_action,
            self.cfg.min_action,
        )
        return action, {}
