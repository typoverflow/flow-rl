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
    init_noise_std: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    entropy_coeff: float
    value_loss_coef: float
    use_clipped_value_loss: bool
    desired_kl: float
    schedule: str
    normalize_advantage: bool
    num_envs: int
    rollout_length: int
    num_minibatches: int
    num_epochs: int
    max_grad_norm: Optional[float]
