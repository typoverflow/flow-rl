from dataclasses import field

import flax.linen as nn
import jax.numpy as jnp

import flowrl.module.initialization as init
from flowrl.types import *


class SimbaBlock(nn.Module):
    hidden_dim: int
    multiplier: int = 4

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        res = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim * 4, kernel_init=init.he_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=init.he_normal())(x)
        return res + x
    
class Simba(nn.Module):
    """
    https://arxiv.org/abs/2410.09754
    """
    hidden_dims: Sequence[int] = field(default_factory=lambda: [])
    output_dim: int = 0
    multiplier: int = 4

    def setup(self):
        assert len(self.hidden_dims) > 0, "hidden_dims must be non-empty"
        for i in range(len(self.hidden_dims)):
            assert self.hidden_dims[i] == self.hidden_dims[0], "All hidden_dims must be the same for simba"

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        # First projection
        x = nn.Dense(self.hidden_dims[0], kernel_init=init.orthogonal_init(1))(x)
        # Residual blocks
        for i, size in enumerate(self.hidden_dims):
            x = SimbaBlock(
                size,
                self.multiplier,
            )(x, training)
        # Final layer norm
        x = nn.LayerNorm()(x)
        # Output projection
        if self.output_dim > 0:
            # No activation here
            x = nn.Dense(self.output_dim, kernel_init=init.orthogonal_init(1))(x)
        return x
