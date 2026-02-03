from dataclasses import dataclass
from typing import List

from ..base import BaseAlgoConfig


@dataclass
class SimbaSACConfig(BaseAlgoConfig):
    name: str
    actor_num_blocks: int
    actor_hidden_dim: int
    actor_lr: float
    actor_wd: float

    critic_num_blocks: int
    critic_hidden_dim: int
    critic_lr: float
    critic_wd: float
    critic_ensemble_size: int

    alpha_lr: float
    alpha_wd: float
    alpha_init_value: float
    target_entropy_scale: float

    ema: float
    discount: float
    num_updates: int
