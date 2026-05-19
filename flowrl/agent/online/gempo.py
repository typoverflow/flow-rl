from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from flowrl.agent.online.dpmd import DPMDAgent, jit_sample_actions, solve_normalizer_exp
from flowrl.config.online.algo.gempo import GeMPOConfig
from flowrl.flow.continuous_ddpm import ContinuousDDPM
from flowrl.functional.ema import ema_update
from flowrl.module.model import Model
from flowrl.types import Batch, Metric, Param, PRNGKey


def solve_normalizer_linear(q: jnp.ndarray, temp: float, negative: float = 0.0):
    q_max = jnp.max(q, axis=-1, keepdims=True)
    q = q - q_max
    num_particles = q.shape[-1]
    q_sorted = jnp.sort(q, axis=-1)[..., ::-1]
    q_cumsum = jnp.cumsum(q_sorted, axis=-1)

    # defining the number of active (non-clipped) particles
    k_indices = jnp.arange(1, num_particles+1).reshape(1, -1)
    excess = q_cumsum - k_indices * q_sorted
    active_mask = excess <= temp * (1 - num_particles * negative)
    k = jnp.sum(active_mask, axis=-1, keepdims=True)

    sum_active = jnp.take_along_axis(q_cumsum, k-1, axis=-1)
    nu = (sum_active - temp * (1 - (num_particles - k) * negative)) / k
    return nu + q_max


def solve_normalizer_square(q: jnp.ndarray, temp: float):
    q_max = jnp.max(q, axis=-1, keepdims=True)
    q = q - q_max
    num_particles = q.shape[-1]
    target_sum = temp ** 2

    q_sorted = jnp.sort(q, axis=-1)[..., ::-1]
    q_cumsum = jnp.cumsum(q_sorted, axis=-1)
    q_squared_cumsum = jnp.cumsum(q_sorted ** 2, axis=-1)

    k_indices = jnp.arange(1, num_particles+1).reshape(1, -1)
    excess = q_squared_cumsum \
            - 2 * q_cumsum * q_sorted \
            + k_indices * (q_sorted ** 2)
    active_mask = excess <= target_sum
    k = jnp.maximum(jnp.sum(active_mask, axis=-1, keepdims=True), 1)

    S1 = jnp.take_along_axis(q_cumsum, k-1, axis=-1)
    S2 = jnp.take_along_axis(q_squared_cumsum, k-1, axis=-1)
    delta = S1**2 - k * S2 + k * target_sum
    nu = (S1 - jnp.sqrt(jnp.maximum(delta, 0.0))) / k
    return nu + q_max


@partial(jax.jit, static_argnames=("discount", "target_kl", "num_particles", "ema", "reweight", "additive_noise", "negative_bound"))
def jit_update_gempo(
    rng: PRNGKey,
    actor: ContinuousDDPM,
    actor_target: ContinuousDDPM,
    critic: Model,
    critic_target: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    reweight: str,
    num_particles: int,
    target_kl: float,
    ema: float,
    additive_noise: float,
    negative_bound: float,
) -> Tuple[PRNGKey, ContinuousDDPM, Model, Model, jnp.ndarray, jnp.ndarray, Metric]:

    # split RNG upfront to remove false sequential dependencies,
    # allowing XLA to parallelize independent sampling calls
    rng, critic_sample_rng, actor_sample_rng, noise_rng, add_noise_rng = jax.random.split(rng, 5)

    # update critic (independent of actor sampling below)
    _, next_action = jit_sample_actions(
        critic_sample_rng,
        actor,
        critic_target,
        batch.next_obs,
        training=False,
        num_samples=1,
        best_of_n=False,
    )
    next_action = next_action[:, 0]
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

    # update actor (independent of critic update above)
    _, action_batch = jit_sample_actions(
        actor_sample_rng,
        actor_target,
        critic,
        batch.obs,
        training=False,
        num_samples=num_particles,
        best_of_n=False,
    )
    noise = jax.random.normal(noise_rng, action_batch.shape)
    action_batch = action_batch + noise * additive_noise
    q_batch = jax.vmap(
        critic,
        in_axes=(None, 1),
        out_axes=2,
    )(batch.obs, action_batch)
    q_batch = q_batch.mean(axis=0).squeeze(-1)

    if reweight == "exp":
        nu = solve_normalizer_exp(q_batch, temp())
        weights = jnp.exp((q_batch - nu) / temp())
    elif reweight == "linear":
        nu = solve_normalizer_linear(q_batch, temp(), negative=negative_bound/num_particles)
        weights = jnp.maximum((q_batch - nu) / temp(), negative_bound/num_particles)
    elif reweight == "square":
        nu = solve_normalizer_square(q_batch, temp())
        weights = jnp.maximum((q_batch - nu) / temp(), 0) ** 2
    else:
        raise ValueError(f"Invalid reweighting method: {reweight}")
    ent_weights = jnp.maximum(weights, 1e-6)
    ent_weights = ent_weights / ent_weights.sum(axis=-1, keepdims=True)
    entropy = - jnp.sum(ent_weights * jnp.log(ent_weights+1e-6), axis=-1)
    weights = weights * num_particles

    _, at, t, eps = actor.add_noise(add_noise_rng, action_batch)

    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        eps_pred = actor.apply(
            {"params": actor_params},
            at,
            t,
            condition=jnp.broadcast_to(batch.obs[:, jnp.newaxis, :], (batch.obs.shape[0], num_particles, batch.obs.shape[-1])),
            training=True,
            rngs={"dropout": dropout_rng},
        )
        loss = jnp.clip((eps_pred - eps) ** 2, a_max=5.0)
        loss = (weights[..., jnp.newaxis] * loss).mean()
        return loss, {
            "loss/actor_loss": loss,
            "misc/weights": weights.mean(),
            "misc/weights_std": weights.std(0).mean(),
            "misc/weights_max": weights.max(0).mean(),
            "misc/weights_min": weights.min(0).mean(),
            "misc/entropy": entropy.mean(),
            "misc/batch_action_std": action_batch.std(axis=1).mean(),
        }
    new_actor, actor_metrics = actor.apply_gradient(actor_loss_fn)

    def temp_loss_fn(temp_value: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        t = temp.apply({"params": temp_value}, rngs={"dropout": dropout_rng})
        loss = t * (
            target_kl + entropy.mean() - jnp.log(num_particles)
        )
        return loss, {
            "loss/temp_loss": loss,
            "misc/temp": t,
        }

    new_temp, temp_metrics = temp.apply_gradient(temp_loss_fn)

    # update the target networks
    new_critic_target = ema_update(new_critic, critic_target, ema)
    new_actor_target = actor_target
    return rng, new_actor, new_actor_target, new_critic, new_critic_target, new_temp, {
        **critic_metrics,
        **actor_metrics,
        **temp_metrics,
    }


class GeMPOAgent(DPMDAgent):
    """
    Generalized Mirror Policy Optimization (GeMPO).

    Extends DPMD by supporting exp / linear / square reweighting schemes.
    """
    name = "GeMPOAgent"
    model_names = ["actor", "critic", "actor_target", "critic_target", "temp"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: GeMPOConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.actor, self.actor_target, self.critic, self.critic_target, self.temp, metrics = jit_update_gempo(
            self.rng,
            self.actor,
            self.actor_target,
            self.critic,
            self.critic_target,
            self.temp,
            batch,
            discount=self.cfg.discount,
            reweight=self.cfg.reweight,
            num_particles=self.cfg.num_particles,
            target_kl=self.cfg.target_kl,
            ema=self.cfg.ema,
            additive_noise=self.cfg.additive_noise,
            negative_bound=self.cfg.negative_bound,
        )
        if self._n_training_steps % self.cfg.old_policy_update_interval == 0:
            self.actor_target = ema_update(self.actor, self.actor_target, 1.0)
        self._n_training_steps += 1
        return metrics
