from typing import List, Optional
from .base import BaseAlgoConfig
from dataclasses import dataclass, field

@dataclass
class IQLConfig(BaseAlgoConfig):
    """Configuration class for the IQL (Implicit Q-Learning) algorithm."""
    name: str = "iql"

    discount: float = 0.99

    tau: float = 0.005 # soft target update
    expectile: float = 0.7
    beta: float = 3.0 # inverse temperature for awr

    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    value_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    critic_ensemble_size: int = 2
    dropout: Optional[float] = None
    layer_norm: bool = False

    deterministic_actor: bool = False
    conditional_logstd: bool = False
    policy_logstd_min: float = -5.0

    actor_lr: float = 3e-4
    value_lr: float = 3e-4
    critic_lr: float = 3e-4
    clip_grad_norm: Optional[float] = None
    lr_decay_steps: Optional[int] = None
    opt_decay_schedule: str = "cosine"

    min_action: float = -1.0
    max_action: float = 1.0
