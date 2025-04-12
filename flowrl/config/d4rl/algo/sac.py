from .base import BaseAlgoConfig
from dataclasses import dataclass, field
from typing import Any, List, Optional

@dataclass
class SACConfig(BaseAlgoConfig):
    """Configuration class for the Soft Actor-Critic (SAC) algorithm."""
    name: str = "sac"

    discount: float = 0.99
    alpha: float = 0.2
    tau: float = 0.005
    auto_entropy: bool = True
    target_update_freq: int = 1

    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4