from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING

from .base import BaseAlgoConfig


@dataclass
class BDPODiffusionTrainConfig():
    lr: float
    lr_decay_steps: int
    clip_grad_norm: float
    ema: float
    ema_every: float

@dataclass
class BDPODiffusionConfig():
    actor: BDPODiffusionTrainConfig
    behavior: BDPODiffusionTrainConfig
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

@dataclass
class BDPOCriticConfig():
    discount: float
    q_target: str
    maxQ: bool
    ensemble_size: int
    rho: float
    eta: float
    hidden_dims: List[int]
    output_nodes: int
    v_min: float
    v_max: float

    lr: float
    lr_decay_steps: int
    clip_grad_norm: float
    layer_norm: bool
    ema: float
    ema_every: int
    steps: int
    num_samples: int
    solver: str
    update_ratio: int

@dataclass
class CategoricalBDPOConfig(BaseAlgoConfig):
    name: str
    warmup_steps: int
    temperature: Optional[float] # None is uniform, 0.0 is greedy
    diffusion: BDPODiffusionConfig
    critic: BDPOCriticConfig
