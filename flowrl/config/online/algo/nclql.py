from dataclasses import dataclass
from typing import List

from .base import BaseAlgoConfig


@dataclass
class AnnealedLangevinDynamicsConfig:
    steps: int
    levels: int
    w: float
    sigma_max: float
    sigma_min: float
    step_lr: float
    q_grad_norm: bool
    clip_sampler: bool
    x_min: float
    x_max: float


@dataclass
class NCLQLConfig(BaseAlgoConfig):
    name: str
    critic_hidden_dims: List[int]
    critic_activation: str
    critic_ensemble_size: int
    time_dim: int
    critic_lr: float
    reward_scale: float
    num_samples: int
    discount: float
    ema: float
    ald: AnnealedLangevinDynamicsConfig
