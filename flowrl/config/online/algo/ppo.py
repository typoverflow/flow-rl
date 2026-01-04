from dataclasses import dataclass
from typing import List

from .base import BaseAlgoConfig


@dataclass
class PPOConfig(BaseAlgoConfig):
    """Configuration for PPO algorithm."""
    name: str = "PPO"

    # Learning rates
    learning_rate: float = 3e-4

    # Network architecture
    actor_hidden_dims: List[int] = None
    critic_hidden_dims: List[int] = None
    layer_norm: bool = False

    # PPO specific parameters
    clip_coef: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    clip_vloss: bool = True
    max_grad_norm: float = 0.5

    def __post_init__(self):
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [64, 64]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [64, 64]
