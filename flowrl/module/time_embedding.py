import jax.numpy as jnp
import flax.linen as nn

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
    
class LearnablePositionalEmbedding(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        w = self.param('kernel', nn.initializers.normal(0.2),
                        (self.output_dim // 2, x.shape[-1]), jnp.float32)
        f = 2 * jnp.pi * x @ w.T
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)

class SinusoidalEmbedding(nn.Module):
    output_dim: int
    max_positions: int = 10000

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        half_dim = self.output_dim // 2
        f = jnp.log(self.max_positions) / (half_dim - 1)
        f = jnp.exp(jnp.arange(half_dim) * -f)
        f = x * f
        return jnp.concatenate([jnp.sin(f), jnp.cos(f)], axis=-1)

class FourierEmbedding(nn.Module):
    output_dim: int
    scale: float = 16

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        freqs = self.param('kernel', nn.initializers.normal(self.scale),
                           (self.output_dim // 2), jnp.float32)
        emb = jnp.einsum('...i,j->...ij', x, (2 * jnp.pi * freqs))
        emb = jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], axis=-1)
        return emb

SUPPORTED_TIMESTEP_EMBEDDING = {
    'positional': PositionalEmbedding,
    'learnable_positional': LearnablePositionalEmbedding,
    'sinusoidal': SinusoidalEmbedding,
    'fourier': FourierEmbedding,
}
