import jax
import jax.numpy as jnp

from flowrl.dataset.base import Batch
from flowrl.types import *


def ema_udpate(
    src: Model,
    tgt: Model,
    ema_rate: float,
) -> Model:
    new_params = jax.tree_util.tree_map(
        lambda p, tp: p * ema_rate + tp * (1 - ema_rate),
        src.params,
        tgt.params,
    )
    return tgt.replace(params=new_params)
