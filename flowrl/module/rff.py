
import flax.linen as nn
import jax.numpy as jnp
import jax

from flowrl.types import *
from .mlp import MLP

class RffLayer(nn.Module):
    feature_dim: int
    rff_dim: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert self.rff_dim % 2 == 0, "rff_dim must be even"
        half = self.rff_dim // 2

        if self.learnable:
            x = MLP(hidden_dims=[], output_dim=half, layer_norm=False, activation=nn.relu)(x)
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
        x = nn.LayerNorm()(x)

        if self.linear:
            feat = x
        else:
            feat = RffLayer(self.feature_dim, self.rff_dim, learnable=True)(
                x
            )

        y = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            layer_norm=True,
            activation=nn.elu,
        )(feat, training=training)
        return y


# implement the double-q logic, foundin EnsembleCritic, use vmap
class RffDoubleQ(nn.Module):
    net1: RffReward
    net2: RffReward
    feature_dim: int
    hidden_dims: Sequence[int]
    linear: bool | None = None
    rff_dim: int | None = None

    @nn.compact
    def __call__(self, x) -> jnp.ndarray:
        # combine this??
        q1 = RffReward(
            self.feature_dim, self.hidden_dims, self.linear, self.rff_dim
        )(x)
        q2 = RffReward(
            self.feature_dim, self.hidden_dims, self.linear, self.rff_dim
        )(x)

        return jnp.stack([q1, q2], axis=0)