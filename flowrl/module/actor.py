import distrax
import flax.linen as nn
import jax.numpy as jnp

from flowrl.types import *
from flowrl.utils.distribution import TanhMultivariateNormalDiag

from .mlp import MLP


class DeterministicActor(nn.Module):
    backbone: nn.Module
    obs_dim: int
    action_dim: int

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        x = self.backbone(obs, training)
        x = MLP(output_dim=self.action_dim)(x)
        return x


class SquashedDeterministicActor(DeterministicActor):
    backbone: nn.Module
    obs_dim: int
    action_dim: int

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        return jnp.tanh(super().__call__(obs, training))


class GaussianActor(nn.Module):
    backbone: nn.Module
    obs_dim: int
    action_dim: int
    conditional_logstd: bool = False
    logstd_min: float = -20.0
    logstd_max: float = 2.0

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        x = self.backbone(obs, training)
        if self.conditional_logstd:
            mean_logstd = MLP(output_dim=2*self.action_dim)(x)
            mean, logstd = jnp.split(mean_logstd, 2, axis=-1)
        else:
            mean = MLP(output_dim=self.action_dim)(x)
            logstd = self.param("logstd", nn.initializers.zeros, (self.action_dim,))
        logstd = jnp.clip(logstd, self.logstd_min, self.logstd_max)
        distribution = distrax.MultivariateNormalDiag(mean, jnp.exp(logstd))
        return distribution


class SquashedGaussianActor(GaussianActor):
    backbone: nn.Module
    obs_dim: int
    action_dim: int
    conditional_logstd: bool = False
    logstd_min: float = -20.0
    logstd_max: float = 2.0

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        training: bool = False,
    ) -> distrax.Distribution:
        x = self.backbone(obs, training)
        if self.conditional_logstd:
            mean_logstd = MLP(output_dim=2*self.action_dim)(x)
            mean, logstd = jnp.split(mean_logstd, 2, axis=-1)
        else:
            mean = MLP(output_dim=self.action_dim)(x)
            logstd = self.param("logstd", nn.initializers.zeros, (self.action_dim,))
        logstd = jnp.clip(logstd, self.logstd_min, self.logstd_max)
        distribution = TanhMultivariateNormalDiag(mean, jnp.exp(logstd))
        return distribution
