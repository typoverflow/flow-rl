import jax
import jax.numpy as jnp
from flowrl.module.model import Model


def ema_udpate(
    src: Model,
    tgt: Model,
    ema_rate: float,
) -> Model:
    new_params = jax.tree_util.tree_map(
        lambda p, tp: p * ema_rate + tp * (1 - ema_rate),
        src.state.params,
        tgt.state.params,
    )
    return tgt.replace(state=tgt.state.replace(params=new_params))
