from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING

from .base import BaseAlgoConfig


@dataclass
class DACDiffusionConfig():
    noise_schedule: str
    resnet: bool
    dropout: Optional[float]
    layer_norm: bool
    time_dim: int
    mlp_hidden_dims: List[int]
    resnet_hidden_dims: List[int]
    solver: str
    steps: int
    clip_sampler: bool
    x_min: float
    x_max: float
    lr: float
    lr_decay_steps: int
    clip_grad_norm: float
    ema: float
    ema_every: int

@dataclass
class DACCriticConfig():
    discount: float
    q_target: str
    maxQ: bool
    ensemble_size: int
    rho: float
    hidden_dims: List[int]

    lr: float
    lr_decay_steps: int
    clip_grad_norm: float
    layer_norm: bool
    ema: float
    ema_every: int
    num_samples: int

@dataclass
class DACConfig(BaseAlgoConfig):
    name: str
    temperature: Optional[float] # None is uniform, 0.0 is greedy
    diffusion: DACDiffusionConfig
    critic: DACCriticConfig

    start_actor_ema: int
    eta: float
    eta_min: float
    eta_max: float
    eta_lr: float
    eta_threshold: float
