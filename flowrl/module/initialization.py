import flax.linen as nn
from flax.linen.initializers import lecun_normal
from jax import numpy as jnp

from flowrl.types import *


def orthogonal_init(scale: Optional[float] = None):
    if scale is None:
        scale = jnp.sqrt(2)
    return nn.initializers.orthogonal(scale)


def uniform_init(scale_final=None):
    if scale_final is not None:
        return nn.initializers.xavier_uniform(scale_final)
    return nn.initializers.xavier_uniform()


default_init = orthogonal_init
