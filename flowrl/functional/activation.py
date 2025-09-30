import flax.linen as nn
import jax
import jax.numpy as jnp


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


# still jitable with branch condition?
def softplus_beta(x, beta=1.0):
    threshold = 20.0
    x_beta = beta * x
    return jnp.where(
        x_beta > threshold, x, (1.0 / beta) * jnp.log(1.0 + jnp.exp(x_beta))
    )
