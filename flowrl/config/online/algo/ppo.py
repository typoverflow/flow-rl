from dataclasses import dataclass
from typing import List, Optional

from .base import BaseAlgoConfig


@dataclass
class PPOConfig(BaseAlgoConfig):
    name: str
    backbone_cls: str
    actor_hidden_dims: List[int]
    critic_hidden_dims: List[int]
    activation: str
    actor_lr: float
    critic_lr: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    entropy_coeff: float
    reward_scaling: float
    normalize_advantage: bool
    num_envs: int
    rollout_length: int
    num_minibatches: int
    num_epochs: int
    batch_size: int
    clip_grad_norm: Optional[float]
