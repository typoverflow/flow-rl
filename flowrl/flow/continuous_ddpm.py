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
    solver: str = field(pytree_node=False, default="ddpm")
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
        solver: str="ddpm",
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
            solver=solver,
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
        solver: str=None,
        steps: Optional[int]=None,
        t_schedule_n: Optional[float]=None,
        params: Optional[Param]=None,
    ) -> Tuple[PRNGKey, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:

        steps = steps or self.steps
        t_schedule_n = t_schedule_n or self.t_schedule_n
        solver = solver or self.solver

        ts = quad_t_schedule(steps, n=t_schedule_n, tmin=self.t_diffusion[0], tmax=self.t_diffusion[1])
        alpha_hats = self.noise_schedule_func(ts)[0] ** 2
        alphas = alpha_hats[1:] / alpha_hats[:-1]
        alphas = jnp.concat([jnp.ones((1, )), alphas], axis=0)
        betas = 1 - alphas

        # Precompute log-SNR for DPM-Solver++(2M)
        if solver == "dpm++2m":
            lambdas = jnp.log(jnp.sqrt(alpha_hats) / jnp.sqrt(jnp.maximum(1 - alpha_hats, 1e-12)))

        # Precompute midpoint noise schedule for DPM-Solver-2
        if solver == "dpm2":
            midpoints = (ts[:-1] + ts[1:]) / 2  # midpoints[j] = midpoint between ts[j] and ts[j+1]
            mid_alphas, mid_sigmas = self.noise_schedule_func(midpoints)

        t_proto = jnp.ones((*xT.shape[:-1], 1), dtype=jnp.int32)

        def _model_fn(xt, t, dropout_rng_):
            if training:
                return self.apply(
                    {"params": params}, xt, t, condition=condition, training=training, rngs={"dropout": dropout_rng_}
                )
            else:
                return self(xt, t, condition=condition, training=training)

        def fn(input_tuple, i):
            rng_, xt, prev_x0_hat, prev_h = input_tuple
            rng_, dropout_rng_, key_ = jax.random.split(rng_, 3)
            input_t = t_proto * ts[i]
            eps_theta = _model_fn(xt, input_t, dropout_rng_)

            x0_hat = (xt - jnp.sqrt(1 - alpha_hats[i]) * eps_theta) / jnp.sqrt(alpha_hats[i])
            x0_hat = jnp.clip(x0_hat, self.x_min, self.x_max) if self.clip_sampler else x0_hat

            h = jnp.zeros(())

            if solver == "ddpm":
                mean_coef1 = jnp.sqrt(alpha_hats[i-1]) * betas[i] / (1 - alpha_hats[i])
                mean_coef2 = jnp.sqrt(alphas[i]) * (1 - alpha_hats[i-1]) / (1 - alpha_hats[i])
                xt_1 = mean_coef1 * x0_hat + mean_coef2 * xt
                xt_1 += (i>1) * jnp.sqrt(betas[i]) * jax.random.normal(key_, xt_1.shape)

            elif solver == "ddim":
                xt_1 = jnp.sqrt(alpha_hats[i-1]) * x0_hat + jnp.sqrt(1 - alpha_hats[i-1]) * eps_theta

            elif solver == "dpm2":
                # DPM-Solver-2: midpoint method (2 NFE per step, 2nd order)
                # midpoints[i-1] is the midpoint between ts[i-1] and ts[i]
                input_s = t_proto * midpoints[i-1]
                alpha_s = mid_alphas[i-1]
                sigma_s = mid_sigmas[i-1]

                # First-order step to midpoint
                x_s = alpha_s * x0_hat + sigma_s * eps_theta

                # Second model evaluation at midpoint
                rng_, dropout_rng2_ = jax.random.split(rng_)
                eps_s = _model_fn(x_s, input_s, dropout_rng2_)

                # Full step from ORIGINAL xt using midpoint's eps estimate
                x0_hat_corrected = (xt - jnp.sqrt(1 - alpha_hats[i]) * eps_s) / jnp.sqrt(alpha_hats[i])
                x0_hat_corrected = jnp.clip(x0_hat_corrected, self.x_min, self.x_max) if self.clip_sampler else x0_hat_corrected
                xt_1 = jnp.sqrt(alpha_hats[i-1]) * x0_hat_corrected + jnp.sqrt(1 - alpha_hats[i-1]) * eps_s

            elif solver == "dpm++2m":
                # DPM-Solver++(2M): multistep 2nd order (1 NFE per step)
                h = lambdas[i-1] - lambdas[i]
                sigma_curr = jnp.sqrt(1 - alpha_hats[i])
                sigma_next = jnp.sqrt(1 - alpha_hats[i-1])
                alpha_next = jnp.sqrt(alpha_hats[i-1])

                # Corrected data prediction (falls back to 1st order when prev_h == 0)
                r = prev_h / jnp.maximum(h, 1e-8)
                d = jnp.where(
                    prev_h > 1e-8,
                    (1 + 0.5 / r) * x0_hat - (0.5 / r) * prev_x0_hat,
                    x0_hat,
                )
                xt_1 = (sigma_next / sigma_curr) * xt - alpha_next * jnp.expm1(-h) * d

            else:
                raise NotImplementedError(f"Unsupported solver: {solver}")

            return (rng_, xt_1, x0_hat, h), (xt, eps_theta)

        init_carry = (rng, xT, jnp.zeros_like(xT), jnp.zeros(()))
        output, history = jax.lax.scan(fn, init_carry, jnp.arange(steps, 0, -1), unroll=True)
        rng, action, _, _ = output
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
