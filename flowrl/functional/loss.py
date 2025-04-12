import jax
import jax.numpy as jnp

from flowrl.types import *


def expectile_regression(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    expectile: float,
) -> jnp.ndarray:
    diff = target - pred
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)

def quantile_regression(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    quantile: float,
) -> jnp.ndarray:
    diff = target - pred
    weight = jnp.where(diff > 0, quantile, (1 - quantile))
    return weight * jnp.abs(diff)

def gumbel_regression(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    diff = (target - pred) / temperature
    diff = jnp.clip(diff, None, 5.0)
    max_diff = jnp.max(diff, axis=0)
    max_diff = jnp.where(max_diff < -1.0, -1.0, max_diff)
    max_diff = jax.lax.stop_gradient(max_diff)
    loss = jnp.exp(diff - max_diff) + jnp.exp(-max_diff) * pred / temperature
    return loss
