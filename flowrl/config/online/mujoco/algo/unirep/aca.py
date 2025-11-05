from dataclasses import dataclass
from typing import List

from ..base import BaseAlgoConfig


@dataclass
class ACADiffusionConfig:
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
class ACAConfig(BaseAlgoConfig):
    name: str
    target_update_freq: int
    feature_dim: int
    rff_dim: int
    critic_hidden_dims: List[int]
    reward_hidden_dims: List[int]
    phi_hidden_dims: List[int]
    mu_hidden_dims: List[int]
    ctrl_coef: float
    reward_coef: float
    critic_coef: float
    critic_activation: str
    back_critic_grad: bool
    feature_lr: float
    critic_lr: float
    discount: float
    num_samples: int
    ema: float
    feature_ema: float
    clip_grad_norm: float | None
    temp: float
    diffusion: ACADiffusionConfig

    num_noises: int
    linear: bool
    ranking: bool
