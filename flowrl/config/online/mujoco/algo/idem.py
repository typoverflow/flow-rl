from dataclasses import dataclass
from typing import List

from .base import BaseAlgoConfig


@dataclass
class IDEMDiffusionConfig:
    time_dim: int
    mlp_hidden_dims: List[int]
    lr: float
    end_lr: float
    lr_decay_steps: int | None
    lr_decay_begin: int
    steps: int
    clip_sampler: bool
    x_min: float
    x_max: float
    solver: str


@dataclass
class IDEMConfig(BaseAlgoConfig):
    name: str
    critic_hidden_dims: List[int]
    critic_lr: float
    discount: float
    num_samples: int
    num_reverse_samples: int
    ema: float
    temp: float
    diffusion: IDEMDiffusionConfig
