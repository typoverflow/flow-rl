from functools import partial
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.offline.algo.fql import FQLConfig
from flowrl.flow.cnf import (
    ContinuousNormalizingFlow,
    FlowBackbone,
    jit_update_flow_matching,
)
from flowrl.functional.ema import ema_update
from flowrl.module.model import Model
from flowrl.types import *

from .network import MLP, EnsembleCritic


@partial(jax.jit, static_argnames=("max_action", "min_action", "act_dim"))
def jit_sample_action(
    rng: PRNGKey,
    actor_onestep: Model,
    obs: jnp.ndarray,
    act_dim: int,
    max_action: float,
    min_action: float,
):
    B, _ = obs.shape
    x0 = jax.random.normal(rng, (B, act_dim))
    action = actor_onestep(jnp.concat([obs, x0], axis=-1))
    action = jnp.clip(action, min_action, max_action)
    return action

def update_critic(
    rng: PRNGKey,
    critic: Model,
    critic_target: Model,
    actor_onestep: Model,
    batch: Batch,
    q_agg: str,
    discount: float,
    max_action: float,
    min_action: float,
) -> Tuple[Model, Metric]:
    rng, sample_rng = jax.random.split(rng)
    next_action = jit_sample_action(
        sample_rng,
        actor_onestep,
        batch.next_obs,
        batch.action.shape[-1],
        max_action,
        min_action,
    )
    next_q = critic_target(batch.next_obs, next_action)
    if q_agg == "mean":
        next_q = jnp.mean(next_q, axis=0)
    elif q_agg == "min":
        next_q = jnp.min(next_q, axis=0)
    else:
        raise ValueError(f"Unknown q_agg: {q_agg}")
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
    rng: PRNGKey,
    actor_onestep: Model,
    actor_bc: ContinuousNormalizingFlow,
    critic: Model,
    batch: Batch,
    alpha: float,
    normalize_q_loss: bool,
    max_action: float,
    min_action: float,
) -> Tuple[Model, Metric]:
    noise_rng, bc_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, (batch.obs.shape[0], actor_bc.x_dim))
    _, bc_action, _ = actor_bc.sample(bc_rng, noise, batch.obs)
    bc_action = jnp.clip(bc_action, min_action, max_action)

    def onestep_loss_fn(onestep_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        pred = actor_onestep.apply(
            {"params": onestep_params},
            jnp.concat([batch.obs, noise], axis=-1),
            training=True,
            rngs={"dropout": dropout_rng},
        )
        distill_loss = ((pred - bc_action) ** 2).mean()
        pred = jnp.clip(pred, min_action, max_action)
        qs = critic(batch.obs, pred)
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
    new_actor_onestep, metrics = actor_onestep.apply_gradient(onestep_loss_fn)
    return new_actor_onestep, metrics

@partial(jax.jit, static_argnames=("alpha", "normalize_q_loss", "discount", "tau", "q_agg", "max_action", "min_action"))
def jit_update_fql(
    rng: PRNGKey,
    actor_bc: ContinuousNormalizingFlow,
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
    rng, noise_rng, critic_rng, bc_rng, onestep_rng = jax.random.split(rng, 5)
    new_critic, critic_metrics = update_critic(
        critic_rng,
        critic,
        critic_target,
        actor_onestep,
        batch,
        q_agg,
        discount,
        max_action,
        min_action,
    )
    # update bc flow
    _, new_actor_bc, bc_metrics = jit_update_flow_matching(
        bc_rng,
        actor_bc,
        jax.random.normal(noise_rng, (batch.obs.shape[0], actor_bc.x_dim)),
        batch.action,
        batch.obs,
    )
    # update onestep actor
    new_actor_onestep, onestep_metrics = update_onestep_actor(
        onestep_rng,
        actor_onestep,
        actor_bc,
        critic,
        batch,
        alpha,
        normalize_q_loss,
        max_action,
        min_action,
    )
    # update target critic
    new_critic_target = ema_update(new_critic, critic_target, tau)
    return rng, new_actor_bc, new_actor_onestep, new_critic, new_critic_target, {
        **bc_metrics,
        **onestep_metrics,
        **critic_metrics,
    }


class FQLAgent(BaseAgent):
    """
    Flow Q-Learning (FQL) agent.
    """
    name = "FQLAgent"
    model_names = ["critic", "target_critic", "actor_bc", "actor_onestep"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: FQLConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, bc_key, critic_key, onestep_key = jax.random.split(self.rng, 4)

        flow_def = FlowBackbone(
            vel_predictor=MLP(
                hidden_dims=cfg.actor_hidden_dims,
                layer_norm=cfg.actor_layer_norm,
                output_dim=act_dim,
                activation=nn.gelu,
            ),
        )

        self.actor_bc = ContinuousNormalizingFlow.create(
            flow_def,
            bc_key,
            inputs=(jnp.ones((1, self.act_dim)), jnp.ones((1, 1)), jnp.ones((1, self.obs_dim))),
            x_dim=act_dim,
            steps=cfg.flow_steps,
            clip_sampler=False,
            optimizer=optax.adam(learning_rate=cfg.lr),
        )

        self.actor_onestep = Model.create(
            MLP(
                hidden_dims=cfg.actor_hidden_dims,
                layer_norm=cfg.actor_layer_norm,
                output_dim=act_dim,
                activation=nn.gelu
            ),
            onestep_key,
            inputs=jnp.concat([jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))], axis=-1),
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
        self.rng, self.actor_bc, self.actor_onestep, self.critic, self.critic_target, metrics = jit_update_fql(
            self.rng,
            self.actor_bc,
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
        self.rng, sample_key = jax.random.split(self.rng)
        action = jit_sample_action(
            sample_key,
            self.actor_onestep,
            obs,
            self.act_dim,
            self.cfg.max_action,
            self.cfg.min_action,
        )
        return action, {}
