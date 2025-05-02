from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING

from .base import BaseAlgoConfig


@dataclass
class DQLDiffusionConfig():
    noise_schedule: str
    time_dim: int
    hidden_dims: List[int]
    solver: str # TBD
    steps: int
    clip_sampler: bool
    x_min: float
    x_max: float
    ema: float
    ema_every: int

@dataclass
class DQLCriticConfig():
    discount: float
    q_target: str
    maxQ: bool
    ensemble_size: int
    rho: float
    hidden_dims: List[int]
    ema: float
    ema_every: int
    num_samples: int

@dataclass
class DQLConfig(BaseAlgoConfig):
    name: str
    temperature: Optional[float] # None is uniform, 0.0 is greedy
    diffusion: DQLDiffusionConfig
    critic: DQLCriticConfig

    start_actor_ema: int
    eta: float
    grad_norm: float
    lr: float
    lr_decay_steps: int
