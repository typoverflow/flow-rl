import flax.linen as nn
from jax import numpy as jnp

from flowrl.types import *


def orthogonal_init(scale: Optional[float] = None):
    if scale is None:
        scale = jnp.sqrt(2)
    return nn.initializers.orthogonal(scale)  # , nn.initializers.normal(scale)


def uniform_init(scale_final=None):
    if scale_final is not None:
        return nn.initializers.xavier_uniform(scale_final)
    return nn.initializers.xavier_uniform()


default_init = uniform_init
