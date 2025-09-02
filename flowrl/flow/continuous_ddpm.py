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

# ======= Noise Network =======

ContinuousDDPMBackbone = DDPMBackbone


@dataclass
class ContinuousDDPM(Model):
    state: TrainState
    dropout_rng: PRNGKey = field(pytree_node=True)
    x_dim: int = field(pytree_node=False, default=None)
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
        noise_schedule: str,
        noise_schedule_params: Optional[Dict]=None,
        clip_sampler: bool=False,
        x_min: Optional[float]=None,
        x_max: Optional[float]=None,
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
            clip_sampler=clip_sampler,
            x_min=x_min,
            x_max=x_max,
            t_diffusion=t_diffusion,
            noise_schedule_func=noise_schedule_func,
        )

    @partial(jax.jit, static_argnames=())
    def add_noise(self, rng, x0):
        rng, t_rng, noise_rng = jax.random.split(rng, 3)
        t = jax.random.uniform(t_rng, (*x0.shape[:-1], 1), self.t_diffusion[0], self.t_diffusion[1])
        eps = jax.random.normal(noise_rng, x0.shape)

        alpha, sigma = self.noise_schedule_func(t)
        xt = alpha * x0 + sigma * eps
        return rng, xt, t, eps

    @partial(jax.jit, static_argnames=("training", "solver", "steps", "t_schedule"))
    def sample(
        self,
        rng: PRNGKey,
        xT: jnp.ndarray,
        condition: Optional[jnp.ndarray]=None,
        training: bool=False,
        solver: str="ddpm",
        steps: int=1000,
        t_schedule: str | Callable = "linear",
        params: Optional[Param]=None,
    ) -> Tuple[PRNGKey, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:

        if isinstance(t_schedule, str):
            t_schedule_func = {
                "linear": linear_t_schedule,
                "cosine": cosine_t_schedule,
            }.get(t_schedule)
        elif isinstance(t_schedule, Callable):
            t_schedule_func = t_schedule
        else:
            raise ValueError(f"Unsupported t_schedule: {t_schedule}")

        ts = t_schedule_func(steps)
        alphas, sigmas = self.noise_schedule_func(ts)
        logSNRs = jnp.log(alphas / sigmas)
        stds = sigmas[:-1] / sigmas[1:] * jnp.sqrt(1 - (alphas[1:] / alphas[:-1]) ** 2)
        stds = jnp.concat([jnp.zeros((1, )), stds], axis=0)

        t_proto = jnp.ones((*xT.shape[:-1], 1), dtype=jnp.int32)

        def fn(input_tuple, t):
            rng_, xt = input_tuple
            rng_, dropout_rng_, key_ = jax.random.split(rng_, 3)
            input_t = t_proto * t
            if training:
                eps_theta = self.apply(
                    {"params": params}, xt, input_t, condition=condition, training=training, rngs={"dropout": dropout_rng_}
                )
            else:
                eps_theta = self(xt, input_t, condition=condition, training=training)

            if solver == "ddpm":
                xt_1 = (
                    (alphas[t-1] / alphas[t]) * (xt - sigmas[t] * eps_theta) +
                    jnp.sqrt(sigmas[t-1]**2 - stds[i] ** 2 + 1e-8) * eps_theta
                )
                xt_1 += (t > 1) * (stds[t] * jax.random.normal(key_, xt_1.shape))
            elif solver == "ddim":
                xt_1 = (alphas[t - 1] * ((xt_1 - sigmas[t] * eps_theta) / alphas[t]) + sigmas[t - 1] * eps_theta)
            else:
                raise NotImplementedError(f"Unsupported solver: {solver}")

            xt_1 = jnp.clip(xt_1, self.x_min, self.x_max)
            return (rng_, xt_1), (xt, eps_theta)

        output, history = jax.lax.scan(fn, (rng, xT), ts, unroll=True)
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
