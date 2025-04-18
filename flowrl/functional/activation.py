import flax.linen as nn
import jax
import jax.numpy as jnp


def mish(x):
    return x * jnp.tanh(nn.softplus(x))
