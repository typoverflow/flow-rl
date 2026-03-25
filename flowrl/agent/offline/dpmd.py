from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.offline.algo.dpmd import OfflineDPMDConfig
from flowrl.flow.continuous_ddpm import ContinuousDDPM, ContinuousDDPMBackbone
from flowrl.functional.activation import get_activation
from flowrl.functional.ema import ema_update
from flowrl.functional.loss import expectile_regression
from flowrl.module.critic import Ensemblize, ScalarCritic
from flowrl.module.misc import PositiveTunableCoefficient
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.time_embedding import LearnableFourierEmbedding
from flowrl.types import Batch, Metric, Param, PRNGKey


# ── reweighting normalizers (identical to online DPMD) ──────────────────────

def solve_normalizer_exp(q: jnp.ndarray, temp: float):
    nu = temp * jax.nn.logsumexp(q / temp, axis=-1, keepdims=True)
    return nu

def solve_normalizer_linear(q: jnp.ndarray, temp: float, negative: float = 0.0):
    num_particles = q.shape[-1]
    B = temp * negative
    target_sum = temp * 1 - B
    q_sorted = jnp.sort(q, axis=-1)[..., ::-1]
    q_cumsum = jnp.cumsum(q_sorted, axis=-1)

    k_indices = jnp.arange(1, num_particles + 1).reshape(1, -1)
    excess = q_cumsum - k_indices * q_sorted
    active_mask = excess <= target_sum
    k = jnp.sum(active_mask, axis=-1, keepdims=True)

    sum_active = jnp.take_along_axis(q_cumsum, k - 1, axis=-1)
    nu = (sum_active - target_sum) / k
    nu = nu - B
    return nu

def solve_normalizer_square(q: jnp.ndarray, temp: float):
    num_particles = q.shape[-1]
    target_sum = temp ** 2

    q_sorted = jnp.sort(q, axis=-1)[..., ::-1]
    q_cumsum = jnp.cumsum(q_sorted, axis=-1)
    q_squared_cumsum = jnp.cumsum(q_sorted ** 2, axis=-1)

    k_indices = jnp.arange(1, num_particles + 1).reshape(1, -1)
    excess = q_squared_cumsum \
            - 2 * q_cumsum * q_sorted \
            + k_indices * (q_sorted ** 2)
    active_mask = excess <= target_sum
    k = jnp.maximum(jnp.sum(active_mask, axis=-1, keepdims=True), 1)

    S1 = jnp.take_along_axis(q_cumsum, k - 1, axis=-1)
    S2 = jnp.take_along_axis(q_squared_cumsum, k - 1, axis=-1)
    delta = S1 ** 2 - k * S2 + k * target_sum
    nu = (S1 - jnp.sqrt(jnp.maximum(delta, 0.0))) / k
    return nu


# ── JIT helpers ─────────────────────────────────────────────────────────────

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

    obs_repeat = obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
    xT = jax.random.normal(xT_rng, (*obs_repeat.shape[:-1], actor.x_dim))
    rng, actions, _ = actor.sample(rng, xT, obs_repeat, training)
    if best_of_n:
        qs = critic(obs_repeat, actions)
        qs = qs.min(axis=0).reshape(B, num_samples)
        best_idx = qs.argmax(axis=-1)
        actions = actions.reshape(B, num_samples, -1)[jnp.arange(B), best_idx]
    return rng, actions


@jax.jit
def jit_update_behavior(
    rng: PRNGKey,
    behavior: ContinuousDDPM,
    batch: Batch,
) -> Tuple[PRNGKey, ContinuousDDPM, Metric]:
    rng, at, t, eps = behavior.add_noise(rng, batch.action)

    def bc_loss_fn(params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        eps_pred = behavior.apply(
            {"params": params},
            at,
            t,
            condition=batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        loss = ((eps_pred - eps) ** 2).mean()
        return loss, {"loss/behavior_bc_loss": loss}

    new_behavior, metrics = behavior.apply_gradient(bc_loss_fn)
    return rng, new_behavior, metrics


@partial(jax.jit, static_argnames=("expectile",))
def jit_update_value(
    value: Model,
    critic_target: Model,
    batch: Batch,
    expectile: float,
) -> Tuple[Model, Metric]:
    qs = critic_target(batch.obs, batch.action)
    qs = qs.min(axis=0)

    def value_loss_fn(value_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        v = value.apply(
            {"params": value_params},
            batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        value_loss = expectile_regression(v, qs, expectile).mean()
        return value_loss, {
            "loss/value_loss": value_loss,
            "misc/v_mean": v.mean(),
        }

    new_value, metrics = value.apply_gradient(value_loss_fn)
    return new_value, metrics


@partial(jax.jit, static_argnames=("discount",))
def jit_update_critic(
    critic: Model,
    value: Model,
    batch: Batch,
    discount: float,
) -> Tuple[Model, Metric]:
    target_q = batch.reward + discount * (1 - batch.terminal) * value(batch.next_obs)

    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        qs = critic.apply(
            {"params": critic_params},
            batch.obs,
            batch.action,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        critic_loss = ((qs - target_q[jnp.newaxis, :]) ** 2).sum(0).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": qs.mean(),
            "misc/reward": batch.reward.mean(),
        }

    new_critic, metrics = critic.apply_gradient(critic_loss_fn)
    return new_critic, metrics


@partial(jax.jit, static_argnames=("num_particles", "target_kl", "reweight", "additive_noise"))
def jit_update_actor(
    rng: PRNGKey,
    actor: ContinuousDDPM,
    behavior: ContinuousDDPM,
    critic: Model,
    temp: Model,
    batch: Batch,
    num_particles: int,
    target_kl: float,
    reweight: str,
    additive_noise: float,
) -> Tuple[PRNGKey, ContinuousDDPM, Model, Metric]:
    # sample particles from behavior policy
    rng, action_batch = jit_sample_actions(
        rng,
        behavior,
        critic,
        batch.obs,
        training=False,
        num_samples=num_particles,
        best_of_n=False,
    )
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, action_batch.shape)
    action_batch = action_batch + noise * additive_noise

    # compute Q-values for all particles
    q_batch = jax.vmap(
        critic,
        in_axes=(None, 1),
        out_axes=2,
    )(batch.obs, action_batch)
    q_batch = q_batch.mean(axis=0).squeeze(-1)  # (B, num_particles)

    # reweight
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
    entropy = -jnp.sum(weights * jnp.log(weights + 1e-6), axis=-1)
    weights = weights * num_particles

    # add noise to action_batch for diffusion loss
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

    # temperature update
    def temp_loss_fn(temp_value: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        t_val = temp.apply({"params": temp_value}, rngs={"dropout": dropout_rng})
        loss = t_val * (target_kl + entropy.mean() - jnp.log(num_particles))
        return loss, {
            "loss/temp_loss": loss,
            "misc/temp": t_val,
        }

    new_temp, temp_metrics = temp.apply_gradient(temp_loss_fn)

    return rng, new_actor, new_temp, {**actor_metrics, **temp_metrics}


# ── Agent class ─────────────────────────────────────────────────────────────

class OfflineDPMDAgent(BaseAgent):
    """
    Offline Diffusion Policy Mirror Descent (DPMD)

    Combines IQL-style value learning with DPMD particle reweighting.
    A behavior diffusion model is trained in parallel to generate
    action particles for policy optimization.
    """
    name = "OfflineDPMDAgent"
    model_names = ["actor", "behavior", "critic", "critic_target", "value", "temp"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: OfflineDPMDConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_rng, behavior_rng, critic_rng, value_rng, temp_rng = jax.random.split(self.rng, 6)

        actor_activation = get_activation(cfg.diffusion.activation)

        # shared backbone definition for actor and behavior
        def make_backbone():
            return ContinuousDDPMBackbone(
                noise_predictor=MLP(
                    hidden_dims=cfg.diffusion.hidden_dims,
                    output_dim=act_dim,
                    activation=actor_activation,
                ),
                time_embedding=LearnableFourierEmbedding(
                    output_dim=cfg.diffusion.time_dim,
                ),
                cond_embedding=MLP(
                    hidden_dims=(128, 128),
                    activation=actor_activation,
                ),
            )

        ddpm_kwargs = dict(
            x_dim=act_dim,
            steps=cfg.diffusion.steps,
            noise_schedule="cosine",
            noise_schedule_params={},
            clip_sampler=cfg.diffusion.clip_sampler,
            x_min=cfg.diffusion.x_min,
            x_max=cfg.diffusion.x_max,
            t_schedule_n=1.0,
        )
        init_inputs = (
            jnp.ones((1, act_dim)),
            jnp.zeros((1, 1)),
            jnp.ones((1, obs_dim)),
        )

        # actor lr schedule
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
            network=make_backbone(),
            rng=actor_rng,
            inputs=init_inputs,
            optimizer=optax.adam(learning_rate=actor_lr),
            **ddpm_kwargs,
        )

        # behavior policy lr schedule
        if cfg.behavior_lr_decay_steps is not None:
            behavior_lr = optax.linear_schedule(
                init_value=cfg.behavior_lr,
                end_value=cfg.behavior_end_lr,
                transition_steps=cfg.behavior_lr_decay_steps,
                transition_begin=cfg.behavior_lr_decay_begin,
            )
        else:
            behavior_lr = cfg.behavior_lr

        self.behavior = ContinuousDDPM.create(
            network=make_backbone(),
            rng=behavior_rng,
            inputs=init_inputs,
            optimizer=optax.adam(learning_rate=behavior_lr),
            **ddpm_kwargs,
        )

        # critic (ensemble Q)
        critic_def = Ensemblize(
            base=ScalarCritic(
                backbone=MLP(
                    hidden_dims=cfg.critic.hidden_dims,
                    activation=actor_activation,
                ),
            ),
            ensemble_size=cfg.critic.ensemble_size,
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, obs_dim)), jnp.ones((1, act_dim))),
            optimizer=optax.adam(learning_rate=cfg.critic.lr),
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, obs_dim)), jnp.ones((1, act_dim))),
        )

        # value (IQL-style, obs-only)
        value_def = ScalarCritic(
            backbone=MLP(
                hidden_dims=cfg.value.hidden_dims,
                activation=actor_activation,
            ),
        )
        self.value = Model.create(
            value_def,
            value_rng,
            inputs=(jnp.ones((1, obs_dim)),),
            optimizer=optax.adam(learning_rate=cfg.value.lr),
        )

        # temperature
        self.temp = Model.create(
            PositiveTunableCoefficient(init_value=1.0),
            rng=temp_rng,
            inputs=(),
            optimizer=optax.adam(learning_rate=cfg.temp_lr),
        )

        self._n_training_steps = 0

    def pretrain_step(self, batch: Batch, step: int) -> Metric:
        # pretrain behavior diffusion (BC) and critic/value (IQL, no EMA)
        self.rng, self.behavior, bc_metrics = jit_update_behavior(
            self.rng, self.behavior, batch,
        )
        self.value, value_metrics = jit_update_value(
            self.value, self.critic_target, batch, self.cfg.value.expectile,
        )
        self.critic, critic_metrics = jit_update_critic(
            self.critic, self.value, batch, self.cfg.critic.discount,
        )
        return {**bc_metrics, **value_metrics, **critic_metrics}

    def prepare_training(self):
        # copy critic to critic_target after pretraining
        self.critic_target = ema_update(self.critic, self.critic_target, 1.0)

    def train_step(self, batch: Batch, step: int) -> Metric:
        # 1. behavior diffusion BC (parallel training)
        self.rng, self.behavior, bc_metrics = jit_update_behavior(
            self.rng, self.behavior, batch,
        )

        # 2. value update (IQL expectile regression)
        self.value, value_metrics = jit_update_value(
            self.value, self.critic_target, batch, self.cfg.value.expectile,
        )

        # 3. critic update (IQL TD with V target)
        self.critic, critic_metrics = jit_update_critic(
            self.critic, self.value, batch, self.cfg.critic.discount,
        )

        # 4. actor update (DPMD reweighting with behavior samples)
        self.rng, self.actor, self.temp, actor_metrics = jit_update_actor(
            self.rng,
            self.actor,
            self.behavior,
            self.critic,
            self.temp,
            batch,
            num_particles=self.cfg.num_particles,
            target_kl=self.cfg.target_kl,
            reweight=self.cfg.reweight,
            additive_noise=self.cfg.additive_noise,
        )

        # 5. EMA update critic target
        self.critic_target = ema_update(self.critic, self.critic_target, self.cfg.critic.ema)

        self._n_training_steps += 1
        return {**bc_metrics, **value_metrics, **critic_metrics, **actor_metrics}

    def sample_actions(
        self,
        obs: jnp.ndarray,
        deterministic: bool = True,
        num_samples: int = 1,
    ) -> Tuple[jnp.ndarray, Metric]:
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
