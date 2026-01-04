from dataclasses import field

import flax.linen as nn
import jax.numpy as jnp

from flowrl.module.initialization import default_bias_init, default_kernel_init
from flowrl.types import *


class MLP(nn.Module):
    """FQL put layer norm after activation in their official implementation.
    Reference: https://github.com/seohongpark/fql/blob/719d04417cb0b206deddbc7d761b3e710c3d750c/utils/networks.py#L56
    """
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
            x = nn.Dense(size, kernel_init=default_kernel_init(), bias_init=default_bias_init())(x)
            x = self.activation(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            if self.dropout and self.dropout > 0:
                x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        if self.output_dim > 0:
            x = nn.Dense(self.output_dim, kernel_init=default_kernel_init(), bias_init=default_bias_init())(x)
        return x

class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        if action is None:
            x = obs
        else:
            x = jnp.concatenate([obs, action], axis=-1)
        x = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            activation=self.activation,
            layer_norm=self.layer_norm,
            dropout=self.dropout,
        )(x, training)
        return x


class EnsembleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None
    ensemble_size: int = 2

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.ensemble_size
        )
        x = vmap_critic(
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            layer_norm=self.layer_norm,
            dropout=self.dropout,
        )(obs, action, training)
        return x
