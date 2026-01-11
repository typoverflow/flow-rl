from dataclasses import dataclass
from typing import List

from ..base import BaseAlgoConfig


@dataclass
class BRCConfig(BaseAlgoConfig):
    name: str
    num_bins: int
    v_max: float
    discount: float
    ema: float
    actor_hidden_dim: int
    critic_hidden_dim: int
    critic_ensemble_size: int
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    clip_grad_norm: float | None
