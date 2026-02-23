from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp


def get_activation(activation: str):
    return {
        "relu": jax.nn.relu,
        "elu": jax.nn.elu,
        "mish": mish,
    }[activation]

def mish(x):
    return x * jnp.tanh(nn.softplus(x))


def _softplus_stable(u):
    return jnp.maximum(u, 0) + jnp.log1p(jnp.exp(-jnp.abs(u)))


@partial(jax.jit, static_argnames=("beta", "threshold"))
def softplus_beta(x, beta: float = 1.0, threshold: float = 20.0):
    if beta == 0.0:
        return jnp.maximum(x, 0.0)

    xb = beta * x
    sp = _softplus_stable(xb) / beta
    return jnp.where(xb > threshold, x, sp)

@partial(jax.jit, static_argnames=("group_size"))
def l2_normalize(x, group_size: int | None = None):
    if group_size is None:
        group_size = x.shape[-1]
    assert x.shape[-1] % group_size == 0
    x = x.reshape(x.shape[:-1] + (-1, group_size))
    norm = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    stable_norm = jnp.maximum(norm, 1e-8)
    x = x / stable_norm
    x = x.reshape(x.shape[:-2] + (-1,))
    return x
