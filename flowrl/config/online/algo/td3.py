from dataclasses import dataclass
from typing import List

from .base import BaseAlgoConfig


@dataclass
class TD3Config(BaseAlgoConfig):
    name: str
    backbone_cls: str
    actor_update_freq: int
    target_update_freq: int
    discount: float
    ema: float
    actor_hidden_dims: List[int]
    actor_activation: str
    critic_hidden_dims: List[int]
    critic_activation: str
    critic_ensemble_size: int
    actor_lr: float
    critic_lr: float
    clip_grad_norm: float | None
    target_policy_noise: float
    noise_clip: float
    exploration_noise: float
