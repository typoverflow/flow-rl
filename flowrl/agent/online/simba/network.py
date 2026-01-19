import flax.linen as nn
import jax.numpy as jnp

from flowrl.types import *


def orthogonal_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


def xavier_normal_init():
    return nn.initializers.glorot_normal()


def xavier_uniform_init():
    return nn.initializers.glorot_uniform()


def he_normal_init():
    return nn.initializers.he_normal()


def he_uniform_init():
    return nn.initializers.he_uniform()


class ResidualBlock(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        res = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(
            self.hidden_dim * 4, kernel_init=he_normal_init()
        )(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=he_normal_init())(x)
        return res + x


class SimbaNet(nn.Module):
    num_blocks: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal_init(1)
        )(x)
        for _ in range(self.num_blocks):
            x = ResidualBlock(self.hidden_dim)(x)
        x = nn.LayerNorm()(x)
        return x


class SimbaCritic(nn.Module):
    num_blocks: int
    hidden_dim: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray|None=None, training: bool = False) -> jnp.ndarray:
        if action is not None:
            x = jnp.concatenate([obs, action], axis=-1)
        else:
            x = obs
        x = SimbaNet(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim
        )(x, training)
        x = nn.Dense(
            1, kernel_init=orthogonal_init(1.0)
        )(x)
        return x

class EnsembleSimbaCritic(nn.Module):
    num_blocks: int
    hidden_dim: int
    ensemble_size: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray|None=None, training: bool = False) -> jnp.ndarray:
        vmap_critic = nn.vmap(
            SimbaCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.ensemble_size
        )
        x = vmap_critic(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim
        )(obs, action, training)
        return x

from flowrl.module.rff import RffLayer


class RffSimbaCritic(nn.Module):
    num_blocks: int
    hidden_dim: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray|None=None, training: bool = False) -> jnp.ndarray:
        if action is not None:
            x = jnp.concatenate([obs, action], axis=-1)
        else:
            x = obs
        x = nn.LayerNorm()(x)
        x = RffLayer(1, self.hidden_dim*2, learnable=True)(x)
        for _ in range(self.num_blocks):
            x = ResidualBlock(self.hidden_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(
            1, kernel_init=orthogonal_init(1.0)
        )(x)
        return x

class EnsembleRffSimbaCritic(nn.Module):
    num_blocks: int
    hidden_dim: int
    ensemble_size: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray|None=None, training: bool = False) -> jnp.ndarray:
        vmap_critic = nn.vmap(
            RffSimbaCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.ensemble_size
        )
        x = vmap_critic(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim
        )(obs, action, training)
        return x
