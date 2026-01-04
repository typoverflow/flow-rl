from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING

from .base import BaseAlgoConfig


@dataclass
class IQLConfig(BaseAlgoConfig):
    name: str

    discount: float

    ema: float # soft target update
    expectile: float
    beta: float # inverse temperature for awr

    actor_hidden_dims: List[int]
    value_hidden_dims: List[int]
    critic_hidden_dims: List[int]
    critic_ensemble_size: int
    dropout: Optional[float]
    layer_norm: bool

    deterministic_actor: bool
    conditional_logstd: bool
    policy_logstd_min: float

    actor_lr: float
    value_lr: float
    critic_lr: float
    clip_grad_norm: Optional[float]
    lr_decay_steps: Optional[int]
    opt_decay_schedule: str

    min_action: float
    max_action: float
