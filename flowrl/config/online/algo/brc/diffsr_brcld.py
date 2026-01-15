from dataclasses import dataclass
from typing import List

from ..base import BaseAlgoConfig


@dataclass
class LDConfig:
    steps: int
    schedule: str
    stepsize_init: float
    stepsize_final: float
    stepsize_decay: float
    stepsize_power: float
    noise_scale: float
    grad_clip: float | None
    drift_clip: float | None
    margin_clip: float | None


@dataclass
class DiffSRBRCLDConfig(BaseAlgoConfig):
    name: str
    num_updates: int
    num_bins: int
    v_max: float
    discount: float
    ema: float
    actor_hidden_dim: int
    critic_hidden_dim: int
    critic_ensemble_size: int
    critic_lr: float
    clip_grad_norm: float | None

    num_noises: int
    feature_dim: int
    feature_lr: float
    embed_dim: int
    rff_dim: int
    phi_hidden_dims: List[int]
    mu_hidden_dims: List[int]
    reward_hidden_dims: List[int]
    reward_coef: float

    exploration_noise: float
    num_samples: int
    wd: float
    ld_temp: float
    ld: LDConfig
