from dataclasses import dataclass
from typing import List

from .base import BaseAlgoConfig


@dataclass
class SACConfig(BaseAlgoConfig):
    name: str
    backbone_cls: str
    discount: float
    ema: float
    actor_hidden_dims: List[int]
    actor_activation: str
    critic_hidden_dims: List[int]
    critic_activation: str
    critic_ensemble_size: int
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    clip_grad_norm: float | None
