from functools import partial
from typing import Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass, field
from flax.training.train_state import TrainState

from flowrl.module.model import Model
from flowrl.types import Param, PRNGKey


def inverse_softplus(value: float | jnp.ndarray) -> jnp.ndarray:
    value = jnp.asarray(value)
    return jnp.where(value > 20.0, value, jnp.log(jnp.expm1(value)))


def diagonal_normal_log_prob(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    std: jnp.ndarray,
) -> jnp.ndarray:
    log_two_pi = jnp.log(2.0 * jnp.pi)
    return -0.5 * jnp.sum(
        jnp.square((value - mean) / std)
        + 2.0 * jnp.log(std)
        + log_two_pi,
        axis=-1,
    )


def tanh_log_det_jacobian(value: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(
        2.0 * (jnp.log(2.0) - value - jax.nn.softplus(-2.0 * value)),
        axis=-1,
    )


class DIMEScoreNetwork(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray]
    time_dim: int = 256
    layer_norm: bool = False
    initial_dt: float = 0.1
    initial_friction: float = 1.0
    per_dim_friction: bool = True
    output_clip: float = 1e4
    output_kernel_init: float = 1e-8

    @nn.compact
    def __call__(
        self,
        action: jnp.ndarray,
        obs: jnp.ndarray,
        time: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        self.param(
            "raw_dt",
            lambda key: jnp.full((1,), inverse_softplus(self.initial_dt)),
        )
        friction_dim = self.action_dim if self.per_dim_friction else 1
        self.param(
            "raw_friction",
            lambda key: jnp.full(
                (friction_dim,),
                inverse_softplus(self.initial_friction),
            ),
        )
        phase = self.param(
            "time_phase",
            nn.initializers.zeros,
            (1, self.time_dim),
        )
        coefficients = jnp.linspace(0.1, 100.0, self.time_dim)[None]
        time_features = jnp.concatenate(
            [
                jnp.sin(coefficients * time + phase),
                jnp.cos(coefficients * time + phase),
            ],
            axis=-1,
        )
        time_features = nn.Dense(self.time_dim)(time_features)
        time_features = nn.gelu(time_features)
        time_features = nn.Dense(self.time_dim)(time_features)

        x = jnp.concatenate([action, obs, time_features], axis=-1)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation(x)
        x = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.constant(self.output_kernel_init),
            bias_init=nn.initializers.zeros,
        )(x)
        return jnp.clip(x, -self.output_clip, self.output_clip)


@dataclass
class DIMEFlow(Model):
    """DIME's overdamped diffusion-control policy."""

    state: TrainState
    dropout_rng: PRNGKey = field(pytree_node=True)
    x_dim: int = field(pytree_node=False, default=None)
    steps: int = field(pytree_node=False, default=None)
    prior_std: float = field(pytree_node=False, default=2.5)
    learn_dt: bool = field(pytree_node=False, default=True)
    learn_friction: bool = field(pytree_node=False, default=True)
    schedule_min: float = field(pytree_node=False, default=0.001)
    schedule_offset: float = field(pytree_node=False, default=0.008)
    schedule_power: int = field(pytree_node=False, default=2)

    @classmethod
    def create(
        cls,
        network: nn.Module,
        rng: PRNGKey,
        inputs: Sequence[jnp.ndarray],
        x_dim: int,
        steps: int,
        prior_std: float,
        learn_dt: bool,
        learn_friction: bool,
        schedule_min: float,
        schedule_offset: float,
        schedule_power: int,
        optimizer: Optional[optax.GradientTransformation] = None,
    ) -> "DIMEFlow":
        if steps < 1:
            raise ValueError(f"DIME steps must be positive, got {steps}")
        model = super().create(network, rng, inputs, optimizer)
        return model.replace(
            x_dim=x_dim,
            steps=steps,
            prior_std=prior_std,
            learn_dt=learn_dt,
            learn_friction=learn_friction,
            schedule_min=schedule_min,
            schedule_offset=schedule_offset,
            schedule_power=schedule_power,
        )

    @partial(jax.jit, static_argnames=("training",))
    def sample(
        self,
        rng: PRNGKey,
        obs: jnp.ndarray,
        training: bool = False,
        params: Optional[Param] = None,
    ) -> Tuple[
        PRNGKey,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        params = self.params if params is None else params
        rng, prior_rng, scan_rng = jax.random.split(rng, 3)
        initial = (
            jax.random.normal(
                prior_rng,
                (*obs.shape[:-1], self.x_dim),
            )
            * self.prior_std
        )
        initial_log_prob = diagonal_normal_log_prob(
            initial,
            jnp.zeros_like(initial),
            jnp.full_like(initial, self.prior_std),
        )
        initial_log_ratio = jnp.zeros(obs.shape[:-1], dtype=obs.dtype)

        raw_dt = params["raw_dt"]
        raw_friction = params["raw_friction"]
        dt_value = jax.nn.softplus(raw_dt)
        friction = jax.nn.softplus(raw_friction)
        if not self.learn_dt:
            dt_value = jax.lax.stop_gradient(dt_value)
        if not self.learn_friction:
            friction = jax.lax.stop_gradient(friction)
        schedule_offset = 1.0 + self.schedule_offset

        def integrate(carry, step):
            step_rng, action, log_ratio = carry
            step_rng, noise_rng = jax.random.split(step_rng)
            step_float = step.astype(action.dtype)
            time = jnp.full(
                (*action.shape[:-1], 1),
                step_float,
                dtype=action.dtype,
            )
            remaining = (self.steps - step_float) / self.steps
            dt = dt_value * (
                (1.0 - self.schedule_min)
                * jnp.cos(
                    0.5
                    * jnp.pi
                    * (schedule_offset - remaining)
                    / schedule_offset
                )
                ** self.schedule_power
                + self.schedule_min
            )
            inverse_friction = 1.0 / friction
            eta = dt * inverse_friction
            scale = jnp.sqrt(2.0 * eta)

            prior_drift = -action / (self.prior_std ** 2)
            control = self.apply(
                {"params": params},
                action,
                obs,
                time,
                training=training,
            )
            forward_mean = action + eta * (prior_drift + control)
            next_action = forward_mean + scale * jax.random.normal(
                noise_rng,
                action.shape,
            )

            next_prior_drift = -next_action / (self.prior_std ** 2)
            backward_mean = next_action + eta * next_prior_drift
            forward_log_prob = diagonal_normal_log_prob(
                next_action,
                forward_mean,
                jnp.broadcast_to(scale, action.shape),
            )
            backward_log_prob = diagonal_normal_log_prob(
                action,
                backward_mean,
                jnp.broadcast_to(scale, action.shape),
            )
            log_ratio = (
                log_ratio + backward_log_prob - forward_log_prob
            )
            return (step_rng, next_action, log_ratio), next_action

        (_, raw_action, log_ratio), latents = jax.lax.scan(
            integrate,
            (scan_rng, initial, initial_log_ratio),
            jnp.arange(self.steps),
            unroll=True,
        )
        action = jnp.tanh(raw_action)
        running_cost = -(
            log_ratio + tanh_log_det_jacobian(raw_action)
        )
        stochastic_cost = jnp.zeros_like(running_cost)
        latent_path = jnp.concatenate(
            [initial[jnp.newaxis, ...], latents],
            axis=0,
        )
        latent_path = latent_path.at[-1].set(action)
        return (
            rng,
            action,
            running_cost[..., None],
            stochastic_cost[..., None],
            initial_log_prob[..., None],
            latent_path,
        )
