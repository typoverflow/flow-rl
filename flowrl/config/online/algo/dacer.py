from dataclasses import dataclass
from typing import List

from .base import BaseAlgoConfig


@dataclass
class DACERDiffusionConfig:
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

@dataclass
class DACERConfig(BaseAlgoConfig):
    name: str
    backbone_cls: str
    critic_hidden_dims: List[int]
    critic_activation: str
    critic_ensemble_size: int
    critic_lr: float
    alpha_lr: float
    discount: float
    ema: float
    entropy_num_samples: int
    update_actor_every: int
    update_alpha_every: int
    reward_scale: float
    noise_scaler: float
    diffusion: DACERDiffusionConfig
