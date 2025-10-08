from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.agent.online.qsm import QSMAgent, jit_update_qsm_critic
from flowrl.config.online.mujoco.algo.idem import IDEMConfig
from flowrl.flow.continuous_ddpm import ContinuousDDPM, ContinuousDDPMBackbone
from flowrl.functional.activation import mish
from flowrl.functional.ema import ema_update
from flowrl.module.critic import EnsembleCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.time_embedding import LearnableFourierEmbedding
from flowrl.types import Batch, Metric, Param, PRNGKey

jit_update_idem_critic = jit_update_qsm_critic

@partial(jax.jit, static_argnames=("num_reverse_samples", "temp",))
def jit_update_idem_actor(
    rng: PRNGKey,
    actor: ContinuousDDPM,
    critic_target: Model,
    batch: Batch,
    num_reverse_samples: int,
    temp: float,
) -> Tuple[PRNGKey, ContinuousDDPM, Metric]:
    a0 = batch.action
    obs_repeat = batch.obs[jnp.newaxis, ...].repeat(num_reverse_samples, axis=0)

    rng, tnormal_rng, clipped_rng = jax.random.split(rng, 3)
    rng, at, t, eps = actor.add_noise(rng, a0)
    alpha1, alpha2 = actor.noise_schedule_func(t)
    lower_bound = - 1.0 / alpha2 * at - alpha1 / alpha2
    upper_bound = - 1.0 / alpha2 * at + alpha1 / alpha2
    tnormal_noise = jax.random.truncated_normal(tnormal_rng, lower_bound, upper_bound, (num_reverse_samples, *at.shape))
    normal_noise = jax.random.normal(clipped_rng, (num_reverse_samples, *at.shape))
    normal_noise_clipped = jnp.clip(normal_noise, lower_bound, upper_bound)
    eps_reverse = jnp.where(jnp.isnan(tnormal_noise), normal_noise_clipped, tnormal_noise)
    a0_hat = 1 / alpha1 * at + alpha2 / alpha1 * eps_reverse

    q_value_and_grad_fn = jax.vmap(
        jax.vmap(
            jax.value_and_grad(lambda a, s: critic_target(s, a).min(axis=0).mean()),
        )
    )
    q_value, q_grad = q_value_and_grad_fn(a0_hat, obs_repeat)
    q_grad = q_grad / temp
    weight = jax.nn.softmax(q_value / temp, axis=0)
    eps_estimation = - (alpha2 / alpha1) * jnp.sum(weight[:, :, jnp.newaxis] * q_grad, axis=0)

    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        eps_pred = actor.apply(
            {"params": actor_params},
            at,
            t,
            condition=batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        loss = ((eps_pred - eps_estimation) ** 2).mean()
        return loss, {
            "loss/actor_loss": loss,
            "misc/eps_estimation_l1": jnp.abs(eps_estimation).mean(),
            "misc/weights": weight.mean(),
            "misc/weight_std": weight.std(0).mean(),
            "misc/weight_max": weight.max(0).mean(),
            "misc/weight_min": weight.min(0).mean(),
        }

    new_actor, actor_metrics = actor.apply_gradient(actor_loss_fn)
    return rng, new_actor, actor_metrics


class IDEMAgent(QSMAgent):
    """
    Iterative Denoising Energy Matching (iDEM) Agent.
    """
    name = "IDEMAgent"
    model_names = ["actor", "critic", "critic_target"]

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.critic, self.critic_target, critic_metrics = jit_update_idem_critic(
            self.rng,
            self.actor,
            self.critic,
            self.critic_target,
            batch,
            discount=self.cfg.discount,
            solver=self.cfg.diffusion.solver,
            ema=self.cfg.ema,
        )
        self.rng, self.actor, actor_metrics = jit_update_idem_actor(
            self.rng,
            self.actor,
            self.critic_target,
            batch,
            num_reverse_samples=self.cfg.num_reverse_samples,
            temp=self.cfg.temp,
        )
        self._n_training_steps += 1
        return {**critic_metrics, **actor_metrics}
