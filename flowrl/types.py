import collections
from typing import Any, Callable, Dict, NewType, Optional, Sequence, Tuple, Union

import flax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

PRNGKey = NewType("PRNGKey", jax.Array)
Param = NewType("Param", flax.core.FrozenDict[str, Any])
Shape = NewType("Shape", Sequence[int])
Metric = NewType("Metric", Dict[str, Any])
Batch = collections.namedtuple(
    'Batch',
    ['obs', 'action', 'reward', 'terminal', 'next_obs', 'next_action'],
)

__all__ = ["Batch", "PRNGKey", "Param", "Shape", "Metric", "Optional", "Sequence", "Any", "Dict", "Callable", "Union", "Tuple"]
