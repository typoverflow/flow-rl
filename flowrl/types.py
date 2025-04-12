import collections
from typing import Any, Callable, Dict, NewType, Optional, Sequence, Union

import flax
import jax
import jax.numpy as jnp

PRNGKey = NewType("PRNGKey", Union[int, jnp.ndarray])
Param = NewType("Param", flax.core.FrozenDict[str, Any])
Shape = NewType("Shape", Sequence[int])
Metric = NewType("Metric", Dict[str, Any])
Batch = collections.namedtuple(
    'Batch',
    ['obs', 'action', 'reward', 'terminal', 'next_obs']
)
Model = NewType("Model", flax.training.train_state.TrainState)

__all__ = ["Batch", "Model", "PRNGKey", "Param", "Shape", "Metric", "Optional", "Sequence", "Any", "Dict", "Callable", "Union"]
