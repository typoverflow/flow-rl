from dataclasses import dataclass
from typing import List

from .base import BaseAlgoConfig


@dataclass
class QVPODiffusionConfig:
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
class QVPOConfig(BaseAlgoConfig):
    name: str
    backbone_cls: str
    critic_hidden_dims: List[int]
    critic_activation: str
    critic_lr: float
    discount: float
    num_behavior_samples: int
    num_evaluate_samples: int
    num_train_samples: int
    reweight: str
    entropy_coef: float
    ema: float
    clip_grad_norm: float
    diffusion: QVPODiffusionConfig
