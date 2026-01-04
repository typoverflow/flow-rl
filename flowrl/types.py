import collections
from typing import Any, Callable, Dict, NewType, Optional, Sequence, Tuple, Union

import flax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from dataclasses import dataclass
from functools import partial

PRNGKey = NewType("PRNGKey", jax.Array)
Param = NewType("Param", flax.core.FrozenDict[str, Any])
Shape = NewType("Shape", Sequence[int])
Metric = NewType("Metric", Dict[str, Any])

@partial(
    jax.tree_util.register_dataclass, 
    data_fields=["obs", "action", "reward", "terminal", "next_obs", "next_action"], 
    meta_fields=[], 
)
@dataclass
class Batch:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    terminal: jnp.ndarray
    next_obs: jnp.ndarray
    next_action: jnp.ndarray

__all__ = ["Batch", "PRNGKey", "Param", "Shape", "Metric", "Optional", "Sequence", "Any", "Dict", "Callable", "Union", "Tuple"]
