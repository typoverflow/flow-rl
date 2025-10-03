import flax.linen as nn
import jax.numpy as jnp
import jax

from flowrl.functional.activation import mish
from flowrl.types import *

from .mlp import MLP


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
            axis_size=self.ensemble_size,
        )
        x = vmap_critic(
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            layer_norm=self.layer_norm,
            dropout=self.dropout,
        )(obs, action, training)
        return x


class CriticT(nn.Module):
    time_embedding: nn.Module
    hidden_dims: Sequence[int]
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None

    @nn.compact
    def __call__(
        self,
        obs: Optional[jnp.ndarray] = None,
        action: Optional[jnp.ndarray] = None,
        t: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        t_ff = self.time_embedding()(t)
        t_ff = MLP(
            hidden_dims=[t_ff.shape[-1], t_ff.shape[-1]],
            activation=mish,
        )(t_ff)
        x = jnp.concatenate(
            [item for item in [obs, action, t_ff] if item is not None], axis=-1
        )
        x = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            activation=self.activation,
            layer_norm=self.layer_norm,
            dropout=self.dropout,
        )(x, training)
        return x


class EnsembleCriticT(nn.Module):
    time_embedding: nn.Module
    hidden_dims: Sequence[int]
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None
    ensemble_size: int = 2

    @nn.compact
    def __call__(
        self,
        obs: Optional[jnp.ndarray] = None,
        action: Optional[jnp.ndarray] = None,
        t: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        vmap_critic_t = nn.vmap(
            CriticT,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.ensemble_size,
        )
        x = vmap_critic_t(
            time_embedding=self.time_embedding,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            layer_norm=self.layer_norm,
            dropout=self.dropout,
        )(obs, action, t, training)
        return x
