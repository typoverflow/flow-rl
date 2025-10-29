import flax.linen as nn
import jax.numpy as jnp

from flowrl.functional.activation import mish
from flowrl.module.mlp import MLP
from flowrl.types import *


class EnergyNet(nn.Module):
    mlp_impl: nn.Module
    hidden_dims: Sequence[int]
    output_dim: int = 1
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None
    cond_embedding: Optional[nn.Module] = None
    time_embedding: Optional[nn.Module] = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        t: Optional[jnp.ndarray] = None,
        condition: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        if condition is not None:
            if self.cond_embedding is not None:
                condition = self.cond_embedding()(condition, training=training)
            else:
                condition = condition
            x = jnp.concatenate([x, condition], axis=-1)
        if self.time_embedding is not None:
            t_ff = self.time_embedding()(t)
            t_ff = MLP(
                hidden_dims=[t_ff.shape[-1], t_ff.shape[-1]],
                activation=mish,
            )(t_ff)
            x = jnp.concatenate([x, t_ff], axis=-1)
        x = self.mlp_impl(
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation=self.activation,
            layer_norm=self.layer_norm,
            dropout=self.dropout,
        )(x, training)
        return x


class EnsembleEnergyNet(nn.Module):
    mlp_impl: nn.Module
    hidden_dims: Sequence[int]
    output_dim: int = 1
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None
    cond_embedding: Optional[nn.Module] = None
    time_embedding: Optional[nn.Module] = None
    ensemble_size: int = 2

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        t: Optional[jnp.ndarray] = None,
        condition: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        vmap_energy_net = nn.vmap(
            EnergyNet,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.ensemble_size
        )
        x = vmap_energy_net(
            mlp_impl=self.mlp_impl,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation=self.activation,
            layer_norm=self.layer_norm,
            dropout=self.dropout,
            cond_embedding=self.cond_embedding,
            time_embedding=self.time_embedding,
        )(x, t, condition, training)
        return x
