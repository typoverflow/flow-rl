from typing import Any, Callable, Dict, NewType, Optional, Sequence, Union

import flax
import jax
import jax.numpy as jnp

PRNGKey = NewType("PRNGKey", Union[int, jnp.ndarray])
Param = NewType("Param", flax.core.FrozenDict[str, Any])
Shape = NewType("Shape", Sequence[int])
Metric = NewType("Metric", Dict[str, Any])

__all__ = ["PRNGKey", "Param", "Shape", "Metric", "Optional", "Sequence", "Any", "Dict", "Callable", "Union"]
