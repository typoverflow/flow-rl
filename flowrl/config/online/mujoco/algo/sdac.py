from dataclasses import dataclass
from typing import List

from .base import BaseAlgoConfig


@dataclass
class SDACDiffusionConfig:
    time_dim: int
    mlp_hidden_dims: List[int]
    lr: float
    end_lr: float
    lr_decay_steps: int | None
    lr_decay_begin: int
    steps: int
    noise_schedule: str
    clip_sampler: bool
    x_min: float
    x_max: float


@dataclass
class SDACConfig(BaseAlgoConfig):
    name: str
    critic_hidden_dims: List[int]
    critic_lr: float
    alpha_lr: float
    target_entropy_scale: float
    discount: float
    num_samples: int
    num_reverse_samples: int
    ema: float
    diffusion: SDACDiffusionConfig
    solver: str
    noise_scaler: float
