from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.struct import PyTreeNode, dataclass, field
from flax.training.train_state import TrainState

from flowrl.functional.activation import mish
from flowrl.functional.ema import ema_update
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.types import *

from .ddpm import DDPMBackbone as EDMBackbone

# ======= Utils ========

def append_dims(x:jnp.ndarray, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (jnp.newaxis,) * dims_to_append]

# ======= EDM sample density ========

def rand_log_normal(rng, shape, loc, scale):
    """Draws samples from a lognormal distribution."""
    normal_samples = jax.random.normal(rng, shape)
    log_normal_samples = jnp.exp(normal_samples * scale + loc)
    return log_normal_samples


def rand_log_logistic(rng, shape, loc, scale, min_value, max_value):
    """Draws samples from an optionally truncated log-logistic distribution."""
    # NOTE: DTQL's official implementation uses float64 here, but jax does not support unless enabling flot64 for all operations
    # min_value = jnp.array(min_value, dtype=jnp.float64)
    # max_value = jnp.array(max_value, dtype=jnp.float64)
    min_cdf = jax.scipy.special.expit((jnp.log(min_value) - loc) / scale)
    max_cdf = jax.scipy.special.expit((jnp.log(max_value) - loc) / scale)
    # u = jax.random.uniform(rng, shape, minval=min_cdf, maxval=max_cdf, dtype=jnp.float64)
    u = jax.random.uniform(rng, shape, minval=min_cdf, maxval=max_cdf)
    log_logistic_samples = jnp.exp(jax.scipy.special.logit(u) * scale + loc)
    return log_logistic_samples

# ======= EDM ========

@dataclass
class EDM(Model):
    state: TrainState
    dropout_rng: PRNGKey = field(pytree_node=True)
    x_dim: int = field(pytree_node=False, default=None)
    steps: int = field(pytree_node=False, default=None)
    clip_sampler: bool = field(pytree_node=False, default=None)
    x_min: float = field(pytree_node=False, default=None)
    x_max: float = field(pytree_node=False, default=None)
    sample_density: Callable = field(pytree_node=False, default=None)
    sigma_data: float = field(pytree_node=False, default=None)
    sigma_min: float = field(pytree_node=False, default=None)
    sigma_max: float = field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        network: nn.Module,
        rng: PRNGKey,
        inputs: Sequence[jnp.ndarray],
        x_dim: int,
        steps: int,
        sigma_sample_density_type: str,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        clip_sampler: bool=False,
        x_min: Optional[float]=None,
        x_max: Optional[float]=None,
        optimizer: Optional[optax.GradientTransformation]=None,
        clip_grad_norm: float=None
    ) -> 'EDM':
        ret = super().create(network, rng, inputs, optimizer, clip_grad_norm)

        if sigma_sample_density_type == 'loglogistic': # used in DTQL
            sample_density = partial(
                rand_log_logistic,
                loc=jnp.log(sigma_data).item(),
                scale=0.5,
                min_value=sigma_min,
                max_value=sigma_max
            )
        else:
            raise ValueError('Unknown sample density type')

        return ret.replace(
            x_dim=x_dim,
            steps=steps,
            clip_sampler=clip_sampler,
            x_min=x_min,
            x_max=x_max,
            sample_density=sample_density,
            sigma_data=sigma_data,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

    def get_diffusion_scalings(self, sigma: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    @partial(jax.jit, static_argnames=("training", "solver"))
    def sample(
        self,
        rng: PRNGKey,
        xT: jnp.ndarray,
        condition: Optional[jnp.ndarray]=None,
        training: bool=False,
        solver: str="ddpm",
        params: Optional[Param]=None,
    ) -> Tuple[PRNGKey, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        raise NotImplementedError("EDM does not support sampling yet.")


# ======= Loss Function ========

def compute_edm_loss(
    rng: PRNGKey,
    edm: EDM,
    x0: jnp.ndarray,
    condition: Optional[jnp.ndarray]=None,
    training: bool=False,
    params: Optional[Param]=None,
    dropout_rng: Optional[PRNGKey]=None,
) -> Tuple[PRNGKey, jnp.ndarray]:
    rng, t_key, noise_key = jax.random.split(rng, 3)
    t = edm.sample_density(t_key, x0.shape[0])
    c_skip, c_out, c_in = [append_dims(c, 2) for c in edm.get_diffusion_scalings(t)]
    t = append_dims(t, x0.ndim)
    noise = jax.random.normal(noise_key, x0.shape)
    x_1 = x0 + noise * t
    t = jnp.log(t) / 4

    if training:
        model_output = edm.apply(
            {"params": params},
            x_1 * c_in,
            t,
            condition=condition,
            training=True,
            rngs={"dropout": dropout_rng},
        )
    else:
        model_output = edm(x_1 * c_in, t, condition=condition)
    denoised_x = c_out * model_output + c_skip * x_1
    if edm.clip_sampler:
        denoised_x = jnp.clip(denoised_x, edm.x_min, edm.x_max)
    loss = (((denoised_x - x0) / c_out) ** 2).mean()
    return rng, loss
