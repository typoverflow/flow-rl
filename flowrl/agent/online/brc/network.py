from typing import Callable, Optional

import distrax
import flax.linen as nn
import jax.numpy as jnp


def default_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

class BroNetBlock(nn.Module):
    hidden_dim: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        res = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        res = nn.LayerNorm()(res)
        res = self.activation(res)
        res = nn.Dense(self.hidden_dim, kernel_init=default_init())(res)
        res = nn.LayerNorm()(res)
        return res + x

class BroNet(nn.Module):
    hidden_dim: int
    num_blocks: int = 1
    output_dim: int = 0
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = self.activation(x)
        for _ in range(self.num_blocks):
            x = BroNetBlock(self.hidden_dim, self.activation)(x)
        if self.output_dim > 0:
            x = nn.Dense(self.output_dim, kernel_init=default_init())(x)
        return x

class BroNetCritic(nn.Module):
    hidden_dim: int
    num_blocks: int = 1
    output_dim: int = 0
    activation: Callable = nn.relu

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        x = jnp.concatenate([obs, action], axis=-1)
        x = BroNet(
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            output_dim=self.output_dim,
            activation=self.activation,
        )(x)
        return x

class EnsembleBroNetCritic(nn.Module):
    hidden_dim: int
    num_blocks: int = 1
    output_dim: int = 0
    activation: Callable = nn.relu
    ensemble_size: int = 2

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        vmap_critic = nn.vmap(
            BroNetCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.ensemble_size
        )
        x = vmap_critic(
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            output_dim=self.output_dim,
            activation=self.activation,
        )(obs, action, training)
        return x
