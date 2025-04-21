from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING

from .base import BaseAlgoConfig


@dataclass
class IVRConfig(BaseAlgoConfig):
    """Configuration class for the IQL (Implicit Q-Learning) algorithm."""
    name: str

    discount: float

    ema: float # soft target update

    actor_hidden_dims: List[int]
    value_hidden_dims: List[int]
    critic_hidden_dims: List[int]
    critic_ensemble_size: int
    layer_norm: bool
    actor_dropout: float
    value_dropout: float

    conditional_logstd: bool
    policy_logstd_min: float

    actor_lr: float
    value_lr: float
    critic_lr: float
    lr_decay_steps: Optional[int]
    opt_decay_schedule: str

    min_action: float
    max_action: float

    alpha: float
    method: str
