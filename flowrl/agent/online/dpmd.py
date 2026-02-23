from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.online.algo.dpmd import DPMDConfig
from flowrl.flow.continuous_ddpm import ContinuousDDPM, ContinuousDDPMBackbone
from flowrl.functional.activation import get_activation
from flowrl.functional.ema import ema_update
from flowrl.module.critic import Ensemblize, ScalarCritic
from flowrl.module.misc import PositiveTunableCoefficient
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.simba import Simba
from flowrl.module.time_embedding import LearnableFourierEmbedding
from flowrl.types import Batch, Metric, Param, PRNGKey


def solve_normalizer_exp(q: jnp.ndarray, temp: float):
    nu = temp * jax.nn.logsumexp(q / temp, axis=-1, keepdims=True)
    return nu

def solve_normalizer_linear(q: jnp.ndarray, temp: float, negative: float=0.0):
    num_particles = q.shape[-1]
    B = temp * negative
    target_sum = temp * 1 - B
    q_sorted = jnp.sort(q, axis=-1)[..., ::-1]
    q_cumsum = jnp.cumsum(q_sorted, axis=-1)

    # defining the number of active particles
    k_indices = jnp.arange(1, num_particles+1).reshape(1, -1)
    excess = q_cumsum - k_indices * q_sorted
    active_mask = excess <= target_sum
    k = jnp.sum(active_mask, axis=-1, keepdims=True)

    sum_active = jnp.take_along_axis(q_cumsum, k-1, axis=-1)
    nu = (sum_active - target_sum) / k
    nu = nu - B
    return nu

def solve_normalizer_square(q: jnp.ndarray, temp: float):
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
    return nu

@partial(jax.jit, static_argnames=("training", "num_samples", "best_of_n"))
def jit_sample_actions(
    rng: PRNGKey,
    actor: ContinuousDDPM,
    critic: Model,
    obs: jnp.ndarray,
    training: bool,
    num_samples: int,
    best_of_n: bool,
) -> Tuple[PRNGKey, jnp.ndarray]:
    assert len(obs.shape) == 2
    B = obs.shape[0]
    rng, xT_rng = jax.random.split(rng)

    # sample
    obs_repeat = obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
    xT = jax.random.normal(xT_rng, (*obs_repeat.shape[:-1], actor.x_dim))
    rng, actions, _ = actor.sample(rng, xT, obs_repeat, training)
    if best_of_n:
        qs = critic(obs_repeat, actions)
        qs = qs.min(axis=0).reshape(B, num_samples)
        best_idx = qs.argmax(axis=-1)
        actions = actions.reshape(B, num_samples, -1)[jnp.arange(B), best_idx]
    return rng, actions

@partial(jax.jit, static_argnames=("discount", "target_kl", "num_particles", "ema", "reweight", "additive_noise"))
def jit_update_dpmd(
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
) -> Tuple[PRNGKey, ContinuousDDPM, Model, Model, jnp.ndarray, jnp.ndarray, Metric]:

    # update critic
    rng, next_action = jit_sample_actions(
        rng,
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

    # update actor
    rng, action_batch = jit_sample_actions(
        rng,
        actor_target,
        critic,
        batch.obs,
        training=False,
        num_samples=num_particles,
        best_of_n=False,
    )
    rng, noise_rng = jax.random.split(rng)
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
        nu = solve_normalizer_linear(q_batch, temp())
        weights = jnp.maximum((q_batch - nu) / temp(), 0)
    elif reweight == "square":
        nu = solve_normalizer_square(q_batch, temp())
        weights = jnp.maximum((q_batch - nu) / temp(), 0) ** 2
    else:
        raise ValueError(f"Invalid reweighting method: {reweight}")
    entropy = - jnp.sum(weights * jnp.log(weights+1e-6), axis=-1)
    weights = weights * num_particles

    rng, at, t, eps = actor.add_noise(rng, action_batch)

    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        eps_pred = actor.apply(
            {"params": actor_params},
            at,
            t,
            condition=batch.obs[:, jnp.newaxis, :].repeat(num_particles, axis=1),
            training=True,
            rngs={"dropout": dropout_rng},
        )
        loss = (weights[..., jnp.newaxis] * (eps_pred - eps) ** 2).mean()
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
    # new_actor_target = ema_update(new_actor, actor_target, ema)
    new_actor_target = actor_target
    return rng, new_actor, new_actor_target, new_critic, new_critic_target, new_temp, {
        **critic_metrics,
        **actor_metrics,
        **temp_metrics,
    }


class DPMDAgent(BaseAgent):
    """
    Diffusion Policy Mirror Descent (DPMD)
    https://arxiv.org/pdf/2502.00361
    """
    name = "DPMDAgent"
    model_names = ["actor", "critic", "actor_target", "critic_target", "temp"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: DPMDConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)

        self.cfg = cfg
        self.rng, actor_rng, critic_rng, temp_rng = jax.random.split(self.rng, 4)

        backbone_cls = {
            "mlp": MLP,
            "simba": Simba,
        }[cfg.backbone_cls]
        actor_activation = get_activation(cfg.diffusion.activation)
        critic_activation = get_activation(cfg.critic_activation)

        # define the actor
        backbone_def = ContinuousDDPMBackbone(
            noise_predictor=backbone_cls(
                hidden_dims=cfg.diffusion.hidden_dims,
                output_dim=act_dim,
                activation=actor_activation,
            ),
            time_embedding=LearnableFourierEmbedding(
                output_dim=cfg.diffusion.time_dim
            ),
            cond_embedding=MLP(
                hidden_dims=(128, 128),
                activation=actor_activation,
            ),
        )
        if cfg.diffusion.lr_decay_steps is not None:
            actor_lr = optax.linear_schedule(
                init_value=cfg.diffusion.lr,
                end_value=cfg.diffusion.end_lr,
                transition_steps=cfg.diffusion.lr_decay_steps,
                transition_begin=cfg.diffusion.lr_decay_begin,
            )
        else:
            actor_lr = cfg.diffusion.lr

        self.actor = ContinuousDDPM.create(
            network=backbone_def,
            rng=actor_rng,
            inputs=(jnp.ones((1, self.act_dim)), jnp.zeros((1, 1)), jnp.ones((1, self.obs_dim)), ),
            x_dim=self.act_dim,
            steps=cfg.diffusion.steps,
            noise_schedule="cosine",
            noise_schedule_params={},
            clip_sampler=cfg.diffusion.clip_sampler,
            x_min=cfg.diffusion.x_min,
            x_max=cfg.diffusion.x_max,
            t_schedule_n=1.0,
            optimizer=optax.adam(learning_rate=actor_lr),
        )
        self.actor_target = ContinuousDDPM.create(
            network=backbone_def,
            rng=actor_rng,
            inputs=(jnp.ones((1, self.act_dim)), jnp.zeros((1, 1)), jnp.ones((1, self.obs_dim)), ),
            x_dim=self.act_dim,
            steps=cfg.diffusion.steps,
            noise_schedule="cosine",
            noise_schedule_params={},
            clip_sampler=cfg.diffusion.clip_sampler,
            x_min=cfg.diffusion.x_min,
            x_max=cfg.diffusion.x_max,
            t_schedule_n=1.0,
        )
        self.temp = Model.create(
            PositiveTunableCoefficient(init_value=1.0),
            rng=temp_rng,
            inputs=(),
            optimizer=optax.adam(learning_rate=cfg.temp_lr),
        )

        critic_def = Ensemblize(
            base=ScalarCritic(
                backbone=backbone_cls(
                    hidden_dims=cfg.critic_hidden_dims,
                    activation=critic_activation,
                ),
            ),
            ensemble_size=2,
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
        )

        # define tracking variables
        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.actor, self.actor_target, self.critic, self.critic_target, self.temp, metrics = jit_update_dpmd(
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
        )
        if self._n_training_steps % self.cfg.old_policy_update_interval == 0:
            self.actor_target = ema_update(self.actor, self.actor_target, 1.0)
        self._n_training_steps += 1
        return metrics

    def sample_actions(
        self,
        obs: jnp.ndarray,
        deterministic: bool = True,
        num_samples: int = 1,
    ) -> Tuple[jnp.ndarray, Metric]:
        # if deterministic is true, sample cfg.num_samples actions and select the best one
        # if not, sample 1 action
        if deterministic:
            num_samples = self.cfg.num_samples
        else:
            num_samples = 1
        self.rng, action = jit_sample_actions(
            self.rng,
            self.actor,
            self.critic,
            obs,
            training=False,
            num_samples=num_samples,
            best_of_n=deterministic,
        )
        if num_samples == 1:
            action = action[:, 0]
        return action, {}
