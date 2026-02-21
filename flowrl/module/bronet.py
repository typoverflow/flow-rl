from dataclasses import field

import flax.linen as nn
import jax.numpy as jnp

import flowrl.module.initialization as init
from flowrl.types import *


class BronetBlock(nn.Module):
    hidden_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray]

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        res = nn.Dense(self.hidden_dim, kernel_init=init.orthogonal_init())(x)
        res = nn.LayerNorm()(res)
        res = self.activation(res)
        res = nn.Dense(self.hidden_dim, kernel_init=init.orthogonal_init())(res)
        res = nn.LayerNorm()(res)
        return res + x

class BroNet(nn.Module):
    """
    https://github.com/naumix/BiggerRegularizedCategorical/blob/a98a6378cfa4e875c0cf31de06801ae8cb8b1ed5/jaxrl/networks.py
    """
    hidden_dims: Sequence[int] = field(default_factory=lambda: [])
    output_dim: int = 0
    activation: Callable = nn.relu

    def setup(self):
        assert len(self.hidden_dims) > 0, "hidden_dims must be non-empty"
        for i in range(len(self.hidden_dims)):
            assert self.hidden_dims[i] == self.hidden_dims[0], "All hidden_dims must be the same for BroNet"

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.hidden_dims, kernel_init=init.orthogonal_init())(x)
        x = nn.LayerNorm()(x)
        x = self.activation(x)
        for i, size in enumerate(self.hidden_dims):
            x = BronetBlock(size, self.activation)(x)
        if self.output_dim > 0:
            x = nn.Dense(self.output_dim, kernel_init=init.orthogonal_init())(x)
        return x