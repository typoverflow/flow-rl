from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from flowrl.agent.online.sdac import SDACAgent
from flowrl.config.online.algo.dpmd import DPMDConfig
from flowrl.flow.continuous_ddpm import ContinuousDDPM
from flowrl.functional.ema import ema_update
from flowrl.module.model import Model
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("discount", "solver", "q_stats_lr", "q_clip", "temp", "ema"))
def jit_update_dpmd(
    rng: PRNGKey,
    actor: ContinuousDDPM,
    critic: Model,
    critic_target: Model,
    q_running_mean: jnp.ndarray,
    q_running_std: jnp.ndarray,
    batch: Batch,
    discount: float,
    solver: str,
    q_stats_lr: float,
    q_clip: float,
    temp: float,
    ema: float,
) -> Tuple[PRNGKey, ContinuousDDPM, Model, Model, jnp.ndarray, jnp.ndarray, Metric]:

    # update critic
    rng, next_xT_rng = jax.random.split(rng)
    next_xT = jax.random.normal(next_xT_rng, (*batch.next_obs.shape[:-1], actor.x_dim))
    rng, next_action, _ = actor.sample(rng, next_xT, batch.next_obs, training=False, solver=solver)
    q_target = critic_target(batch.next_obs, next_action)
    q_target = batch.reward + discount * (1 - batch.terminal) * q_target.min(axis=0)

    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        q = critic.apply(
            {"params": critic_params},
            batch.obs,
            batch.action,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        critic_loss = ((q - q_target[jnp.newaxis, :])**2).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": q.mean(),
            "misc/reward": batch.reward.mean(),
        }

    new_critic, critic_metrics = critic.apply_gradient(critic_loss_fn)

    # update actor
    rng, update_rng = jax.random.split(rng)
    a0 = next_action
    q0 = critic(batch.next_obs, a0).min(axis=0)
    q0_normalized = (q0 - q_running_mean) / (q_running_std + 1e-6)
    q0_normalized = jnp.clip(q0_normalized, -q_clip, q_clip) / temp
    weights = jnp.exp(q0_normalized)

    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        _, at, t, eps = actor.add_noise(update_rng, a0)
        eps_pred = actor.apply(
            {"params": actor_params},
            at,
            t,
            condition=batch.next_obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        loss = (weights * ((eps_pred - eps) ** 2)).mean()
        return loss, {
            "loss/actor_loss": loss,
            "misc/weights": weights.mean(),
            "misc/weight_std": weights.std(0).mean(),
            "misc/weight_max": weights.max(0).mean(),
            "misc/weight_min": weights.min(0).mean(),
        }

    new_actor, actor_metrics = actor.apply_gradient(actor_loss_fn)

    # update stats and critic_target
    new_critic_target = ema_update(new_critic, critic_target, ema)
    q_running_mean = q_running_mean + q_stats_lr * (q0.mean() - q_running_mean)
    q_running_std = q_running_std + q_stats_lr * (q0.std() - q_running_std)

    return rng, new_actor, new_critic, new_critic_target, q_running_mean, q_running_std, {
        **critic_metrics,
        **actor_metrics,
        "misc/q_running_mean": q_running_mean,
        "misc/q_running_std": q_running_std,
    }


class DPMDAgent(SDACAgent):
    """
    Diffusion Policy Mirror Descent (DPMD) agent.
    """
    name = "DPMDAgent"
    model_names = ["actor", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: DPMDConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)

        # create statistics for normalizing
        self.q_clip = cfg.q_clip
        self.q_stats_lr = cfg.q_stats_lr
        self.q_running_mean = jnp.zeros((), dtype=jnp.float32)
        self.q_running_std = jnp.ones((), dtype=jnp.float32)

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.actor, self.critic, self.critic_target, self.q_running_mean, self.q_running_std, metrics = jit_update_dpmd(
            self.rng,
            self.actor,
            self.critic,
            self.critic_target,
            self.q_running_mean,
            self.q_running_std,
            batch,
            discount=self.cfg.discount,
            solver=self.cfg.diffusion.solver,
            q_stats_lr=self.q_stats_lr,
            q_clip=self.q_clip,
            temp=self.cfg.temp,
            ema=self.cfg.ema,
        )
        self._n_training_steps += 1
        return metrics
