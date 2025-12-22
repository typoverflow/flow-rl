from functools import partial
from typing import Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.struct import PyTreeNode, dataclass, field
from flax.training.train_state import TrainState

from flowrl.flow.continuous_ddpm import cosine_noise_schedule, linear_noise_schedule
from flowrl.module.model import Model
from flowrl.types import *

# ======= Langevin Dynamics Sampling =======

@dataclass
class LangevinDynamics(Model):
    state: TrainState
    dropout_rng: PRNGKey = field(pytree_node=True)
    x_dim: int = field(pytree_node=False, default=None)
    grad_prediction: bool = field(pytree_node=False, default=True)
    steps: int = field(pytree_node=False, default=None)
    step_size: float = field(pytree_node=False, default=None)
    noise_scale: float = field(pytree_node=False, default=None)
    clip_sampler: bool = field(pytree_node=False, default=None)
    x_min: float = field(pytree_node=False, default=None)
    x_max: float = field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        network: nn.Module,
        rng: PRNGKey,
        inputs: Sequence[jnp.ndarray],
        x_dim: int,
        grad_prediction: bool = True,
        steps: int = 100,
        step_size: float = 0.01,
        noise_scale: float = 1.0,
        clip_sampler: bool = False,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        optimizer: Optional[optax.GradientTransformation] = None,
        clip_grad_norm: float = None
    ) -> 'LangevinDynamics':
        ret = super().create(network, rng, inputs, optimizer, clip_grad_norm)

        return ret.replace(
            x_dim=x_dim,
            grad_prediction=grad_prediction,
            steps=steps,
            step_size=step_size,
            noise_scale=noise_scale,
            clip_sampler=clip_sampler,
            x_min=x_min,
            x_max=x_max,
        )

    @partial(jax.jit, static_argnames=("training"))
    def compute_grad(
        self,
        x: jnp.ndarray,
        i: int,
        condition: Optional[jnp.ndarray] = None,
        training: bool = False,
        params: Optional[Param] = None,
        dropout_rng: Optional[PRNGKey] = None
    ) -> jnp.ndarray:
        original_shape = x.shape[:-1]
        t = i * jnp.ones((*x.shape[:-1], 1), dtype=jnp.int32)

        x = x.reshape(-1, x.shape[-1])
        t = t.reshape(-1, 1)
        condition = condition.reshape(-1, condition.shape[-1])
        if self.grad_prediction:
            if training:
                grad = self.apply(
                    {"params": params}, x, t, condition=condition, training=training, rngs={"dropout": dropout_rng}
                )
            else:
                grad = self(x, t, condition=condition, training=training)
            energy = jnp.zeros_like((*x.shape[:-1], 1), dtype=jnp.float32)
        else:
            if training:
                energy_and_grad_fn = jax.vmap(jax.value_and_grad(lambda x, t, condition: self.apply(
                    {"params": params}, x, t, condition=condition, training=training, rngs={"dropout": dropout_rng}
                ).mean()))
            else:
                energy_and_grad_fn = jax.vmap(jax.value_and_grad(lambda x, t, condition: self(x, t, condition=condition, training=training).mean()))
            energy, grad = energy_and_grad_fn(x, t, condition)
        return grad.reshape(*original_shape, self.x_dim), energy.reshape(*original_shape, 1)

    @partial(jax.jit, static_argnames=("training", "steps","step_size","noise_scale"))
    def sample(
        self,
        rng: PRNGKey,
        x_init: jnp.ndarray,
        condition: Optional[jnp.ndarray] = None,
        training: bool = False,
        steps: Optional[int] = None,
        step_size: Optional[float] = None,
        noise_scale: Optional[float] = None,
        params: Optional[Param] = None,
    ) -> Tuple[PRNGKey, jnp.ndarray, Optional[jnp.ndarray]]:
        steps = steps or self.steps
        step_size = step_size or self.step_size
        noise_scale = noise_scale or self.noise_scale

        def fn(input_tuple, i):
            rng_, xt = input_tuple
            rng_, noise_rng, dropout_rng_ = jax.random.split(rng_, 3)

            grad, energy = self.compute_grad(xt, i, condition=condition, training=training, params=params, dropout_rng=dropout_rng_)

            xt_1 = xt + step_size * grad
            if self.clip_sampler:
                xt_1 = jnp.clip(xt_1, self.x_min, self.x_max)
            noise = jax.random.normal(noise_rng, xt_1.shape, dtype=jnp.float32)
            xt_1 += (i>1) * jnp.sqrt(2 * step_size * noise_scale) * noise

            return (rng_, xt_1), (xt, grad, energy)

        output, history = jax.lax.scan(fn, (rng, x_init), jnp.arange(steps, 0, -1), unroll=True)
        rng, action = output
        return rng, action, history


@dataclass
class AnnealedLangevinDynamics(LangevinDynamics):
    state: TrainState
    dropout_rng: PRNGKey = field(pytree_node=True)
    x_dim: int = field(pytree_node=False, default=None)
    grad_prediction: bool = field(pytree_node=False, default=True)
    steps: int = field(pytree_node=False, default=None)
    step_size: float = field(pytree_node=False, default=None)
    noise_scale: float = field(pytree_node=False, default=None)
    clip_sampler: bool = field(pytree_node=False, default=None)
    x_min: float = field(pytree_node=False, default=None)
    x_max: float = field(pytree_node=False, default=None)
    t_schedule_n: float = field(pytree_node=False, default=None)
    t_diffusion: Tuple[float, float] = field(pytree_node=False, default=None)
    noise_schedule_func: Callable = field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        network: nn.Module,
        rng: PRNGKey,
        inputs: Sequence[jnp.ndarray],
        x_dim: int,
        grad_prediction: bool,
        steps: int,
        step_size: float,
        noise_scale: float,
        noise_schedule: str,
        noise_schedule_params: Optional[Dict]=None,
        clip_sampler: bool = False,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        t_schedule_n: float=1.0,
        epsilon: float=0.001,
        optimizer: Optional[optax.GradientTransformation]=None,
        clip_grad_norm: float=None
    ) -> 'AnnealedLangevinDynamics':
        ret = super().create(
            network,
            rng,
            inputs,
            x_dim,
            grad_prediction,
            steps,
            step_size,
            noise_scale,
            clip_sampler,
            x_min,
            x_max,
            optimizer,
            clip_grad_norm,
        )

        if noise_schedule_params is None:
            noise_schedule_params = {}
        if noise_schedule == "cosine":
            t_diffusion = [epsilon, 0.9946]
        else:
            t_diffusion = [epsilon, 1.0]
        if noise_schedule == "linear":
            noise_schedule_func = partial(linear_noise_schedule, **noise_schedule_params)
        elif noise_schedule == "cosine":
            noise_schedule_func = partial(cosine_noise_schedule, **noise_schedule_params)
        elif noise_schedule == "none":
            noise_schedule_func = lambda t: (jnp.ones_like(t), jnp.zeros_like(t))
        else:
            raise NotImplementedError(f"Unsupported noise schedule: {noise_schedule}")

        return ret.replace(
            t_schedule_n=t_schedule_n,
            t_diffusion=t_diffusion,
            noise_schedule_func=noise_schedule_func,
        )

    @partial(jax.jit, static_argnames=("training"))
    def compute_grad(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        condition: Optional[jnp.ndarray] = None,
        training: bool = False,
        params: Optional[Param] = None,
        dropout_rng: Optional[PRNGKey] = None
    ) -> jnp.ndarray:
        original_shape = x.shape[:-1]
        t = t * jnp.ones((*x.shape[:-1], 1), dtype=jnp.int32)

        x = x.reshape(-1, x.shape[-1])
        t = t.reshape(-1, 1)
        condition = condition.reshape(-1, condition.shape[-1])
        if self.grad_prediction:
            if training:
                grad = self.apply(
                    {"params": params}, x, t, condition=condition, training=training, rngs={"dropout": dropout_rng}
                )
            else:
                grad = self(x, t, condition=condition, training=training)
            energy = jnp.zeros((*x.shape[:-1], 1), dtype=jnp.float32)
        else:
            if training:
                energy_and_grad_fn = jax.vmap(jax.value_and_grad(lambda x, t, condition: self.apply(
                    {"params": params}, x, t, condition=condition, training=training, rngs={"dropout": dropout_rng}
                ).mean()))
            else:
                energy_and_grad_fn = jax.vmap(jax.value_and_grad(lambda x, t, condition: self(x, t, condition=condition, training=training).mean()))
            energy, grad = energy_and_grad_fn(x, t, condition)
        # alpha, sigma = self.noise_schedule_func(t)
        # grad = alpha * grad - sigma * x
        return grad.reshape(*original_shape, self.x_dim), energy.reshape(*original_shape, 1)

    def add_noise(self, rng: PRNGKey, x: jnp.ndarray) -> Tuple[PRNGKey, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        rng, t_rng, noise_rng = jax.random.split(rng, 3)
        t = jax.random.uniform(t_rng, (*x.shape[:-1], 1), dtype=jnp.float32, minval=self.t_diffusion[0], maxval=self.t_diffusion[1])
        alpha, sigma = self.noise_schedule_func(t)
        eps = jax.random.normal(noise_rng, x.shape, dtype=jnp.float32)
        xt = alpha * x + sigma * eps
        return rng, xt, t, eps

    @partial(jax.jit, static_argnames=("training", "steps","step_size","noise_scale"))
    def sample(
        self,
        rng: PRNGKey,
        x_init: jnp.ndarray,
        condition: Optional[jnp.ndarray] = None,
        training: bool = False,
        steps: Optional[int] = None,
        step_size: Optional[float] = None,
        noise_scale: Optional[float] = None,
        params: Optional[Param] = None,
    ) -> Tuple[PRNGKey, jnp.ndarray, Optional[jnp.ndarray]]:
        steps = steps or self.steps
        # step_size = step_size or self.step_size
        # noise_scale = noise_scale or self.noise_scale
        t_schedule_n = 1.0
        from flowrl.flow.continuous_ddpm import quad_t_schedule
        ts = quad_t_schedule(steps, n=t_schedule_n, tmin=self.t_diffusion[0], tmax=self.t_diffusion[1])
        alpha_hats = self.noise_schedule_func(ts)[0] ** 2
        alphas = alpha_hats[1:] / alpha_hats[:-1]
        alphas = jnp.concat([jnp.ones((1, )), alphas], axis=0)
        betas = 1 - alphas
        alpha1, alpha2 = self.noise_schedule_func(ts)

        t_proto = jnp.ones((*x_init.shape[:-1], 1), dtype=jnp.int32)

        def fn(input_tuple, i):
            rng_, xt = input_tuple
            rng_, dropout_rng_, key_ = jax.random.split(rng_, 3)
            input_t = t_proto * ts[i]

            q_grad, energy = self.compute_grad(xt, ts[i], condition=condition, training=training, params=params, dropout_rng=dropout_rng_)
            eps_theta = q_grad

            x0_hat = (xt - jnp.sqrt(1 - alpha_hats[i]) * eps_theta) / jnp.sqrt(alpha_hats[i])
            x0_hat = jnp.clip(x0_hat, self.x_min, self.x_max) if self.clip_sampler else x0_hat

            mean_coef1 = jnp.sqrt(alpha_hats[i-1]) * betas[i] / (1 - alpha_hats[i])
            mean_coef2 = jnp.sqrt(alphas[i]) * (1 - alpha_hats[i-1]) / (1 - alpha_hats[i])
            xt_1 = mean_coef1 * x0_hat + mean_coef2 * xt
            xt_1 += (i>1) * jnp.sqrt(betas[i]) * jax.random.normal(key_, xt_1.shape)

            return (rng_, xt_1), (xt, eps_theta, energy)

        output, history = jax.lax.scan(fn, (rng, x_init), jnp.arange(steps, 0, -1), unroll=True)
        rng, action = output
        return rng, action, history


from flowrl.flow.continuous_ddpm import ContinuousDDPM, quad_t_schedule


@dataclass
class ContinuousDDPMLD(ContinuousDDPM):
    state: TrainState
    dropout_rng: PRNGKey = field(pytree_node=True)
    x_dim: int = field(pytree_node=False, default=None)
    steps: int = field(pytree_node=False, default=None)
    clip_sampler: bool = field(pytree_node=False, default=None)
    x_min: float = field(pytree_node=False, default=None)
    x_max: float = field(pytree_node=False, default=None)
    t_schedule_n: float = field(pytree_node=False, default=None)
    t_diffusion: Tuple[float, float] = field(pytree_node=False, default=None)
    noise_schedule_func: Callable = field(pytree_node=False, default=None)

    @partial(jax.jit, static_argnames=("model_fn", "training", "solver", "steps", "t_schedule_n"))
    def sample(
        self,
        rng: PRNGKey,
        model_fn: Callable,
        xT: jnp.ndarray,
        condition: Optional[jnp.ndarray] = None,
        training: bool = False,
        solver: str = "ddpm",
        steps: Optional[int] = None,
        t_schedule_n: Optional[float] = None,
        params: Optional[Param] = None,
    ) -> Tuple[PRNGKey, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        steps = steps or self.steps
        t_schedule_n = t_schedule_n or self.t_schedule_n

        ts = quad_t_schedule(steps, n=t_schedule_n, tmin=self.t_diffusion[0], tmax=self.t_diffusion[1])
        alpha_hats = self.noise_schedule_func(ts)[0] ** 2
        alphas = alpha_hats[1:] / alpha_hats[:-1]
        alphas = jnp.concat([jnp.ones((1, )), alphas], axis=0)
        betas = 1 - alphas
        sigmas = self.noise_schedule_func(ts)[1]

        t_proto = jnp.ones((*xT.shape[:-1], 1), dtype=jnp.int32)

        def fn(input_tuple, i):
            rng_, xt = input_tuple
            rng_, dropout_rng_, key_ = jax.random.split(rng_, 3)
            input_t = t_proto * ts[i]

            energy, q_grad = model_fn(xt, input_t, condition=condition)

            if solver == "ddpm":
                eps_theta = - sigmas[i] * q_grad
                x0_hat = (xt - jnp.sqrt(1 - alpha_hats[i]) * eps_theta) / jnp.sqrt(alpha_hats[i])
                x0_hat = jnp.clip(x0_hat, self.x_min, self.x_max) if self.clip_sampler else x0_hat

                mean_coef1 = jnp.sqrt(alpha_hats[i-1]) * betas[i] / (1 - alpha_hats[i])
                mean_coef2 = jnp.sqrt(alphas[i]) * (1 - alpha_hats[i-1]) / (1 - alpha_hats[i])
                xt_1 = mean_coef1 * x0_hat + mean_coef2 * xt
                xt_1 += (i>1) * jnp.sqrt(betas[i]) * jax.random.normal(key_, xt_1.shape)
            else:
                raise NotImplementedError(f"Unsupported solver: {solver}")

            return (rng_, xt_1), (eps_theta, q_grad)

        output, history = jax.lax.scan(fn, (rng, xT), jnp.arange(steps, 0, -1), unroll=True)
        rng, action = output
        return rng, action, history
