from dataclasses import field

import flax.linen as nn
import jax.numpy as jnp

from flowrl.types import *

from .initialization import default_init


class MLP(nn.Module):
    hidden_dims: Sequence[int] = field(default_factory=lambda: [])
    output_dim: int = 0
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation(x)
            if self.dropout and self.dropout > 0:
                x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        if self.output_dim > 0:
            x = nn.Dense(self.output_dim, kernel_init=default_init())(x)
        return x


class ResidualLinear(nn.Module):
    dim: int
    multiplier: int = 4
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        residual = x
        if self.dropout and self.dropout > 0:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.dim * self.multiplier, kernel_init=default_init())(x)
        x = self.activation(x)
        x = nn.Dense(self.dim, kernel_init=default_init())(x)

        if residual.shape != x.shape:
            residual = nn.Dense(self.dim, kernel_init=default_init())(residual)

        return residual + x


class ResidualMLP(nn.Module):
    hidden_dims: Sequence[int] = field(default_factory=lambda: [])
    output_dim: int = 0
    multiplier: int = 4
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        if len(self.hidden_dims) > 0:
            x = nn.Dense(self.hidden_dims[0], kernel_init=default_init())(x)
        for i, size in enumerate(self.hidden_dims):
            x = ResidualLinear(size, self.multiplier, self.activation, self.layer_norm, self.dropout)(x, training)
        if self.output_dim > 0:
            x = self.activation(x)
            x = nn.Dense(self.output_dim, kernel_init=default_init())(x)
        return x
