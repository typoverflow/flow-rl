from typing import List, Optional
from .base import BaseAlgoConfig
from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class FQLConfig(BaseAlgoConfig):
    """Configuration class for the FQL (Flow Q-Learning, https://arxiv.org/abs/2502.02538) algorithm."""
    name: str

    discount: float
    tau: float
    alpha: float # weight of distill loss
    lr: float

    actor_hidden_dims: List[int]
    actor_layer_norm: bool
    flow_steps: int
    
    critic_hidden_dims: List[int]
    critic_layer_norm: bool
    normalize_q_loss: bool
    q_agg: str

    max_action: float
    min_action: float
