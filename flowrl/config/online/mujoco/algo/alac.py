from dataclasses import dataclass
from typing import List

from .base import BaseAlgoConfig


@dataclass
class ALACLangevinDynamicsConfig:
    resnet: bool
    activation: str
    ensemble_size: int
    time_dim: int
    hidden_dims: List[int]
    cond_hidden_dims: List[int]
    steps: int
    step_size: float
    noise_scale: float
    noise_schedule: str
    clip_sampler: bool
    x_min: float
    x_max: float
    epsilon: float
    lr: float
    clip_grad_norm: float | None

@dataclass
class ALACConfig(BaseAlgoConfig):
    name: str
    discount: float
    ema: float
    num_samples: int
    ld: ALACLangevinDynamicsConfig
