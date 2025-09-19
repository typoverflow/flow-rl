import flax.linen as nn
import jax.numpy as jnp

from flowrl.module.mlp import MLP
from flowrl.types import *


class AvgL1Norm(nn.Module):
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        eps: float = 1e-8,
    ) -> jnp.ndarray:
        return x / jnp.clip(jnp.abs(x).mean(axis=-1, keepdims=True), min=eps)

def avg_l1_norm(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return x / jnp.clip(jnp.abs(x).mean(axis=-1, keepdims=True), min=eps)


class TD7Encoder(nn.Module):
    obs_dim: int
    action_dim: int
    embed_dim: int
    hidden_dim: int
    activation: Callable = nn.elu

    def setup(self):
        # State encoder
        self.zs_layers = MLP(
            hidden_dims=[self.hidden_dim, self.hidden_dim],
            output_dim=self.embed_dim,
            activation=self.activation,
        )
        self.zsa_layers = MLP(
            hidden_dims=[self.hidden_dim, self.hidden_dim],
            output_dim=self.embed_dim,
            activation=self.activation,
        )
        self.avg_l1_norm = AvgL1Norm()

    def zs(self, obs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        out = self.zs_layers(obs, training=training)
        out = self.avg_l1_norm(out)
        return out

    def zsa(self, zs: jnp.ndarray, action: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        out = jnp.concatenate([zs, action], axis=-1)
        out = self.zsa_layers(out, training=training)
        return out

    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        zs = self.zs(obs, training=training)
        zsa = self.zsa(zs, action, training=training)
        return zsa


class TD7Actor(nn.Module):
    obs_dim: int
    action_dim: int
    embed_dim: int
    hidden_dim: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, obs: jnp.ndarray, zs: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # State layers
        out = MLP(output_dim=self.hidden_dim)(obs, training=training)
        out = AvgL1Norm()(out)

        out = jnp.concatenate([out, zs], axis=-1)

        # Main layers
        out = MLP(
            hidden_dims=[self.hidden_dim, self.hidden_dim],
            output_dim=self.action_dim,
            activation=self.activation,
        )(out, training=training)
        return jnp.tanh(out)


class TD7Critic(nn.Module):
    obs_dim: int
    action_dim: int
    embed_dim: int
    hidden_dim: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        zsa: jnp.ndarray,
        zs: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        out = jnp.concatenate([obs, action], axis=-1)
        out = MLP(output_dim=self.hidden_dim)(out, training=training)
        out = AvgL1Norm()(out)
        out = jnp.concatenate([out, zsa, zs], axis=-1)
        out = MLP(
            hidden_dims=[self.hidden_dim, self.hidden_dim],
            output_dim=1,
            activation=self.activation,
        )(out, training=training)
        return out


class TD7EnsembleCritic(nn.Module):
    obs_dim: int
    action_dim: int
    embed_dim: int
    hidden_dim: int
    activation: Callable = nn.relu
    ensemble_size: int = 2

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        zsa: jnp.ndarray,
        zs: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        if self.ensemble_size == 1:
            return TD7Critic(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                activation=self.activation
            )(obs, action, zsa, zs, training=training)
        else:
            vmap_critic = nn.vmap(
                TD7Critic,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=None,
                out_axes=0,
                axis_size=self.ensemble_size
            )
            x = vmap_critic(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                activation=self.activation
            )(obs, action, zsa, zs, training)
            return x
