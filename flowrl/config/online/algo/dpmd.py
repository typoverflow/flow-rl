from dataclasses import dataclass
from typing import List

from .base import BaseAlgoConfig


@dataclass
class DPMDDiffusionConfig:
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
class DPMDConfig(BaseAlgoConfig):
    name: str
    critic_hidden_dims: List[int]
    critic_activation: str
    critic_lr: float
    temp_lr: float
    discount: float
    num_samples: int
    num_particles: int
    target_kl: float
    reweight: str
    ema: float
    diffusion: DPMDDiffusionConfig
