from dataclasses import dataclass
from typing import List, Optional

from .base import BaseAlgoConfig


@dataclass
class DPMDDiffusionConfig:
    hidden_dims: List[int]
    activation: str
    time_dim: int
    steps: int
    lr: float
    end_lr: float
    lr_decay_steps: Optional[int]
    lr_decay_begin: int
    clip_sampler: bool
    x_min: float
    x_max: float


@dataclass
class DPMDCriticConfig:
    hidden_dims: List[int]
    ensemble_size: int
    discount: float
    lr: float
    ema: float


@dataclass
class DPMDValueConfig:
    hidden_dims: List[int]
    lr: float
    expectile: float


@dataclass
class OfflineDPMDConfig(BaseAlgoConfig):
    name: str

    # diffusion (shared architecture for actor and behavior policy)
    diffusion: DPMDDiffusionConfig
    critic: DPMDCriticConfig
    value: DPMDValueConfig

    # DPMD reweighting
    num_particles: int
    num_samples: int
    reweight: str
    target_kl: float
    additive_noise: float
    temp_lr: float

    # behavior policy lr (can differ from actor lr)
    behavior_lr: float
    behavior_end_lr: float
    behavior_lr_decay_steps: Optional[int]
    behavior_lr_decay_begin: int
