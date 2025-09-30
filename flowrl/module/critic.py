import flax.linen as nn
import jax.numpy as jnp
import jax
import flax

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


class RffLayer(nn.Module):
    feature_dim: int
    rff_dim: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert self.rff_dim % 2 == 0, "rff_dim must be even"
        half = self.rff_dim // 2

        if self.learnable:
            x = nn.Dense(half, name="layer")(x)
        else:
            noise = self.variable(
                "noise",
                "matrix",
                lambda rng: jax.random.normal(rng, (self.feature_dim, half)),
            )
            x = x @ noise.value
        return jnp.concat([jnp.sin(x), jnp.cos(x)], axis=-1)


class RffReward(nn.Module):
    feature_dim: int
    hidden_dims: list[int]
    linear: bool = False
    rff_dim: Optional[int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = nn.LayerNorm(name="ln")(x)

        if self.linear:
            feat = x
        else:
            feat = RffLayer(self.feature_dim, self.rff_dim, learnable=True, name="rff")(
                x
            )

        y = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            layer_norm=True,
            activation=nn.elu,
            name="net",
        )(feat, training=training)
        return y


class RffDoubleQ(nn.Module):
    net1: RffReward
    net2: RffReward
    feature_dim: int
    hidden_dims: Sequence[int]
    linear: bool | None = None
    rff_dim: int | None = None

    @nn.compact
    def __call__(self, x) -> jnp.ndarray:
        q1 = RffReward(
            self.feature_dim, self.hidden_dims, self.linear, self.rff_dim, name="net1"
        )(x)
        q2 = RffReward(
            self.feature_dim, self.hidden_dims, self.linear, self.rff_dim, name="net2"
        )(x)

        return jnp.stack([q1, q2], axis=0)
