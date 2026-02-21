import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp

import flowrl.module.initialization as init
from flowrl.functional.activation import mish
from flowrl.types import *

from .mlp import MLP


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None
    kernel_init: Initializer = init.default_kernel_init
    bias_init: Initializer = init.default_bias_init

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
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x, training)
        return x


class EnsembleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None
    ensemble_size: int = 2
    kernel_init: Initializer = init.default_kernel_init
    bias_init: Initializer = init.default_bias_init

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
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(obs, action, training)
        return x

class CriticT(nn.Module):
    time_embedding: nn.Module
    hidden_dims: Sequence[int]
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None
    kernel_init: Initializer = init.default_kernel_init
    bias_init: Initializer = init.default_bias_init

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
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(t_ff)
        x = jnp.concatenate([item for item in [obs, action, t_ff] if item is not None], axis=-1)
        x = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            activation=self.activation,
            layer_norm=self.layer_norm,
            dropout=self.dropout,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x, training)
        return x


class EnsembleCriticT(nn.Module):
    time_embedding: nn.Module
    hidden_dims: Sequence[int]
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None
    ensemble_size: int = 2
    kernel_init: Initializer = init.default_kernel_init
    bias_init: Initializer = init.default_bias_init

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
            axis_size=self.ensemble_size
        )
        x = vmap_critic_t(
            time_embedding=self.time_embedding,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            layer_norm=self.layer_norm,
            dropout=self.dropout,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(obs, action, t, training)
        return x


class Ensemblize(nn.Module):
    base_cls: type[nn.Module]
    base_kwargs: Dict[str, Any]
    ensemble_size: int

    @nn.compact
    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        vmap_cls = nn.vmap(
            self.base_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.ensemble_size
        )
        return vmap_cls(**self.base_kwargs)(*args, **kwargs)


class BasicCritic(nn.Module):
    backbone: nn.Module
    kernel_init: Initializer = init.default_kernel_init
    bias_init: Initializer = init.default_bias_init

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if action is None:
            x = obs
        else:
            x = jnp.concatenate([obs, action], axis=-1)
        x = self.backbone(x)
        x = nn.Dense(
            1, kernel_init=self.kernel_init(), bias_init=self.bias_init()
        )(x)
        return x


class BasicCriticWithDiscreteTime(nn.Module):
    backbone: nn.Module
    time_embedding: nn.Module
    kernel_init: Initializer = init.default_kernel_init
    bias_init: Initializer = init.default_bias_init

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        t: jnp.ndarray = None,
        training: bool = False,
    ) -> jnp.ndarray:
        t_ff = self.time_embedding()(t)
        t_ff = MLP(
            hidden_dims=[t_ff.shape[-1]*2, t_ff.shape[-1]],
            activation=mish,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(t_ff)
        x = jnp.concatenate([
            obs, action, t_ff
        ], axis=-1)
        x = self.backbone(x)
        x = nn.Dense(
            1, kernel_init=self.kernel_init(), bias_init=self.bias_init()
        )(x)
        return x


class CriticWithDiscreteTimeFiLM(nn.Module):
    """Critic with time conditioning via AdaLN (Adaptive Layer Norm).
    LayerNorm(use_scale=False, use_bias=False) then (1+gamma)*x + beta from time;
    gamma/beta projection is zero-initialized so training starts without modulation.
    Takes hidden_dims and time_embedding; builds the backbone internally.
    """
    time_embedding: nn.Module
    hidden_dims: Sequence[int]
    activation: Callable = mish
    kernel_init: Initializer = init.default_kernel_init
    bias_init: Initializer = init.default_bias_init

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        t: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        t_emb = self.time_embedding()(t)
        t_emb = MLP(
            output_dim=t_emb.shape[-1],
            hidden_dims=[t_emb.shape[-1] * 2,],
            activation=mish,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(t_emb)

        x = jnp.concatenate([obs, action], axis=-1)
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(
                dim,
                kernel_init=self.kernel_init(),
                bias_init=self.bias_init(),
                name=f"dense_{i}",
            )(x)
            x = nn.LayerNorm(use_scale=False, use_bias=False)(x)
            # Zero-init so at start (1+gamma)*x + beta = x (no modulation)
            params = nn.Dense(
                2 * dim,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
                name=f"adaln_{i}",
            )(t_emb)
            gamma, beta = params[..., :dim], params[..., dim:]
            x = (1.0 + gamma) * x + beta
            x = self.activation(x)
        x = nn.Dense(
            1, kernel_init=self.kernel_init(), bias_init=self.bias_init()
        )(x)
        return x


class GaussianCritic(nn.Module):
    backbone: nn.Module
    kernel_init: Initializer = init.default_kernel_init
    bias_init: Initializer = init.default_bias_init

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if action is None:
            x = obs
        else:
            x = jnp.concatenate([obs, action], axis=-1)
        x = self.backbone(x)
        x = nn.Dense(
            2, kernel_init=self.kernel_init(), bias_init=self.bias_init()
        )(x)
        mean, std = x[..., :1], jax.nn.softplus(x[..., 1:])
        return mean, std
