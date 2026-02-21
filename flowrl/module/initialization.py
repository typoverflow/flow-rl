import flax.linen as nn
import jax
from flax.linen.initializers import lecun_normal, variance_scaling, zeros_init, he_normal
from jax import numpy as jnp

from flowrl.types import *


def orthogonal_init(scale: Optional[float] = None):
    if scale is None:
        scale = jnp.sqrt(2)
    return nn.initializers.orthogonal(scale)

def pytorch_kernel_init():
    return variance_scaling(scale=1/3, mode="fan_in", distribution="uniform")

def pytorch_bias_init():
    return lambda key, shape, dtype=jnp.float32: (jax.random.uniform(key, shape, dtype)*2-1) / jnp.sqrt(shape[0])

default_kernel_init = orthogonal_init
default_bias_init = zeros_init
