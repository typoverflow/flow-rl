from dataclasses import dataclass
from typing import List

from .base import BaseAlgoConfig


@dataclass
class TD3Config(BaseAlgoConfig):
    name: str
    actor_update_freq: int
    target_update_freq: int
    discount: float
    ema: float
    actor_hidden_dims: List[int]
    critic_hidden_dims: List[int]
    activation: str
    critic_ensemble_size: int
    layer_norm: bool
    actor_lr: float
    critic_lr: float
    clip_grad_norm: float | None
    target_policy_noise: float
    noise_clip: float
    exploration_noise: float
