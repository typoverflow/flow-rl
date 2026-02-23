from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, NewType, Optional, Sequence, Tuple, Union

import flax
import jax
import jax.numpy as jnp
from flax.linen.initializers import Initializer
from flax.training.train_state import TrainState

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

@partial(
    jax.tree_util.register_dataclass,
    data_fields=["obs", "actions", "rewards", "truncated", "terminated", "log_probs", "last_obs"],
    meta_fields=[],
)
@dataclass
class RolloutBatch:
    obs: jnp.ndarray          # (T, B, obs_dim)
    actions: jnp.ndarray      # (T, B, act_dim)
    rewards: jnp.ndarray      # (T, B, 1)
    truncated: jnp.ndarray  # (T, B, 1) — 1 if time-limited
    terminated: jnp.ndarray    # (T, B, 1) — 1 if truly done, 0 otherwise
    log_probs: jnp.ndarray    # (T, B, 1)
    last_obs: jnp.ndarray     # (B, obs_dim) — for bootstrap value

__all__ = ["Batch", "RolloutBatch", "PRNGKey", "Param", "Shape", "Metric", "Optional", "Sequence", "Any", "Dict", "Callable", "Union", "Tuple", "Initializer", "TrainState"]
