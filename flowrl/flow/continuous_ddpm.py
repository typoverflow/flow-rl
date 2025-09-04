from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.struct import PyTreeNode, dataclass, field
from flax.training.train_state import TrainState

from flowrl.flow.ddpm import DDPMBackbone
from flowrl.functional.activation import mish
from flowrl.functional.ema import ema_update
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.types import *

# ======= Noise Schedules =======

@partial(jax.jit, static_argnames=("beta0", "beta1"))
def linear_noise_schedule(t, beta0: float=0.1, beta1: float=20.0):
    log_alpha = -(beta1 - beta0) / 4.0 * (t**2) - beta0 / 2.0 * t
    alpha = jnp.exp(log_alpha)
    sigma = jnp.sqrt(1.0 - alpha**2)
    return alpha, sigma

@partial(jax.jit, static_argnames=("s"))
def cosine_noise_schedule(t, s: float = 0.008):
    alpha = jnp.cos(jnp.pi / 2.0 * (t.clip(0., 0.9946) + s) / (1 + s)) / jnp.cos(
        jnp.pi / 2.0 * s / (1 + s))
    sigma = jnp.sqrt(1.0 - alpha**2)
    return alpha, sigma

# ======= t schedules =======

@partial(jax.jit, static_argnames=("T", "n", "tmin", "tmax"))
def quad_t_schedule(T: int, n: float=1.0, tmin: float=1e-3, tmax: float=1.0):
    schedule = (tmax - tmin) * (
        jnp.linspace(0, 1, T+1, dtype=jnp.float32) ** n
    ) + tmin
    return schedule

# ======= Noise Network =======

ContinuousDDPMBackbone = DDPMBackbone


@dataclass
class ContinuousDDPM(Model):
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

    @classmethod
    def create(
        cls,
        network: nn.Module,
        rng: PRNGKey,
        inputs: Sequence[jnp.ndarray],
        x_dim: int,
        steps: int,
        noise_schedule: str,
        noise_schedule_params: Optional[Dict]=None,
        clip_sampler: bool=False,
        x_min: Optional[float]=None,
        x_max: Optional[float]=None,
        t_schedule_n: float=1.0,
        epsilon: float=0.001,
        optimizer: Optional[optax.GradientTransformation]=None,
        clip_grad_norm: float=None
    ) -> 'ContinuousDDPM':
        ret = super().create(network, rng, inputs, optimizer, clip_grad_norm)

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
        else:
            raise NotImplementedError(f"Unsupported noise schedule: {noise_schedule}")

        return ret.replace(
            x_dim=x_dim,
            steps=steps,
            clip_sampler=clip_sampler,
            x_min=x_min,
            x_max=x_max,
            t_schedule_n=t_schedule_n,
            t_diffusion=t_diffusion,
            noise_schedule_func=noise_schedule_func,
        )

    @partial(jax.jit, static_argnames=())
    def add_noise(self, rng, x0):
        rng, t_rng, noise_rng = jax.random.split(rng, 3)
        t = jax.random.uniform(t_rng, (*x0.shape[:-1], 1), dtype=jnp.float32, minval=self.t_diffusion[0], maxval=self.t_diffusion[1])
        eps = jax.random.normal(noise_rng, x0.shape, dtype=jnp.float32)

        alpha, sigma = self.noise_schedule_func(t)
        xt = alpha * x0 + sigma * eps
        return rng, xt, t, eps

    @partial(jax.jit, static_argnames=("training", "solver", "steps", "t_schedule_n"))
    def sample(
        self,
        rng: PRNGKey,
        xT: jnp.ndarray,
        condition: Optional[jnp.ndarray]=None,
        training: bool=False,
        solver: str="ddpm",
        steps: Optional[int]=None,
        t_schedule_n: Optional[float]=None,
        params: Optional[Param]=None,
    ) -> Tuple[PRNGKey, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:

        steps = steps or self.steps
        t_schedule_n = t_schedule_n or self.t_schedule_n

        ts = quad_t_schedule(steps, n=t_schedule_n, tmin=self.t_diffusion[0], tmax=self.t_diffusion[1])
        alpha_hats = self.noise_schedule_func(ts)[0] ** 2
        alphas = alpha_hats[1:] / alpha_hats[:-1]
        alphas = jnp.concat([jnp.ones((1, )), alphas], axis=0)
        betas = 1 - alphas

        t_proto = jnp.ones((*xT.shape[:-1], 1), dtype=jnp.int32)

        def fn(input_tuple, i):
            rng_, xt = input_tuple
            rng_, dropout_rng_, key_ = jax.random.split(rng_, 3)
            input_t = t_proto * ts[i]
            if training:
                eps_theta = self.apply(
                    {"params": params}, xt, input_t, condition=condition, training=training, rngs={"dropout": dropout_rng_}
                )
            else:
                eps_theta = self(xt, input_t, condition=condition, training=training)

            if solver == "ddpm":
                x0_hat = (xt - jnp.sqrt(1 - alpha_hats[i]) * eps_theta) / jnp.sqrt(alpha_hats[i])
                x0_hat = jnp.clip(x0_hat, self.x_min, self.x_max) if self.clip_sampler else x0_hat

                mean_coef1 = jnp.sqrt(alpha_hats[i-1]) * betas[i] / (1 - alpha_hats[i])
                mean_coef2 = jnp.sqrt(alphas[i]) * (1 - alpha_hats[i-1]) / (1 - alpha_hats[i])
                xt_1 = mean_coef1 * x0_hat + mean_coef2 * xt
                xt_1 += (i>1) * jnp.sqrt(betas[i]) * jax.random.normal(key_, xt_1.shape)
            else:
                raise NotImplementedError(f"Unsupported solver: {solver}")

            return (rng_, xt_1), (xt, eps_theta)

        output, history = jax.lax.scan(fn, (rng, xT), jnp.arange(steps, 0, -1), unroll=True)
        rng, action = output
        return rng, action, history


# ======= Update Function =======

@jax.jit
def jit_update_continuous_ddpm(
    rng: PRNGKey,
    model: ContinuousDDPM,
    x0: jnp.ndarray,
    condition: Optional[jnp.ndarray]=None,
) -> Tuple[PRNGKey, ContinuousDDPM, Metric]:
    rng, xt, t, eps = model.add_noise(rng, x0)

    def loss_fn(params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        eps_pred = model.apply(
            {"params": params},
            xt,
            t,
            condition=condition,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        loss = ((eps_pred - eps) ** 2).mean()
        return loss, {
            "ddpm_loss": loss
        }
    new_model, metrics = model.apply_gradient(loss_fn)
    return rng, new_model, metrics
