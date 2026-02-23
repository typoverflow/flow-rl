from dataclasses import dataclass
from typing import List

from .base import BaseAlgoConfig


@dataclass
class DPMDDiffusionConfig:
    backbone_cls: str
    activation: str
    time_dim: int
    hidden_dims: List[int]
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
    backbone_cls: str
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
    old_policy_update_interval: int
    additive_noise: float
    diffusion: DPMDDiffusionConfig
