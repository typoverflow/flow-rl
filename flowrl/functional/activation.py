import flax.linen as nn
import jax
import jax.numpy as jnp

from functools import partial

def mish(x):
    return x * jnp.tanh(nn.softplus(x))


def _softplus_stable(u):
    return jnp.maximum(u,0) + jnp.log1p(jnp.exp(-jnp.abs(u)))

@partial(jax.jit, static_argnames=("beta","threshold"))
def softplus_beta(x, beta: float = 1.0, threshold: float = 20.0):
    if beta == 0.0:
        return jnp.maximum(x, 0.0)
    
    xb = beta * x
    sp = _softplus_stable(xb) / beta
    return jnp.where(xb > threshold, x, sp)