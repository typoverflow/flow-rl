import flax.linen as nn
import jax
import jax.numpy as jnp

from flowrl.module.mlp import MLP
from flowrl.types import *


class RffLayer(nn.Module):
    feature_dim: int
    rff_dim: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert self.rff_dim % 2 == 0, "rff_dim must be even"
        half = self.rff_dim // 2

        if self.learnable:
            x = MLP(hidden_dims=[], output_dim=half)(x)
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
    rff_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = nn.LayerNorm()(x)
        x = RffLayer(self.feature_dim, self.rff_dim, learnable=True)(x)

        x = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            layer_norm=True,
            activation=nn.elu,
        )(x, training=training)
        return x


class RffCritic(RffReward):
    pass


class RffEnsembleCritic(nn.Module):
    feature_dim: int
    hidden_dims: Sequence[int]
    rff_dim: int
    ensemble_size: int = 2

    @nn.compact
    def __call__(self, x) -> jnp.ndarray:
        vmap_rff = nn.vmap(
            RffReward,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.ensemble_size,
        )
        x = vmap_rff(self.feature_dim, self.hidden_dims, self.rff_dim)(x)
        return x
