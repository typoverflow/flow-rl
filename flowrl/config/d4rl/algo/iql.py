from typing import List, Optional
from .base import BaseAlgoConfig
from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class IQLConfig(BaseAlgoConfig):
    """Configuration class for the IQL (Implicit Q-Learning) algorithm."""
    name: str

    discount: float

    tau: float # soft target update
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
