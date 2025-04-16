from typing import List, Optional
from .base import BaseAlgoConfig
from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class IQLConfig(BaseAlgoConfig):
    """Configuration class for the IQL (Implicit Q-Learning) algorithm."""
    name: str = "iql"

    discount: float = MISSING

    tau: float = MISSING # soft target update
    expectile: float = MISSING
    beta: float = MISSING # inverse temperature for awr

    actor_hidden_dims: List[int] = MISSING
    value_hidden_dims: List[int] = MISSING
    critic_hidden_dims: List[int] = MISSING
    critic_ensemble_size: int = MISSING
    dropout: Optional[float] = MISSING
    layer_norm: bool = MISSING

    deterministic_actor: bool = MISSING
    conditional_logstd: bool = MISSING
    policy_logstd_min: float = MISSING

    actor_lr: float = MISSING
    value_lr: float = MISSING
    critic_lr: float = MISSING
    clip_grad_norm: Optional[float] = MISSING
    lr_decay_steps: Optional[int] = MISSING
    opt_decay_schedule: str = MISSING

    min_action: float = MISSING
    max_action: float = MISSING
