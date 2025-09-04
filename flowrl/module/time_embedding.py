import flax.linen as nn
import jax.numpy as jnp
import numpy as np


class PositionalEmbedding(nn.Module):
    output_dim: int
    max_positions: int = 10000

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        half_dim = self.output_dim // 2
        f = jnp.log(self.max_positions) / (half_dim - 1)
        f = jnp.exp(jnp.arange(half_dim) * -f)
        f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class LearnableFourierEmbedding(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        w = self.param('kernel', nn.initializers.normal(0.2),
                        (self.output_dim // 2, x.shape[-1]), jnp.float32)
        f = 2 * jnp.pi * x @ w.T
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


SUPPORTED_TIMESTEP_EMBEDDING = {
    "positional": PositionalEmbedding,
    "learnable_fourier": LearnableFourierEmbedding,
}
