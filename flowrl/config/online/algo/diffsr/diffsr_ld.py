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
class DiffSRLDConfig(BaseAlgoConfig):
    name: str
    actor_update_freq: int
    target_update_freq: int
    discount: float
    ema: float
    # critic_hidden_dims: List[int]
    critic_activation: str # not used
    critic_ensemble_size: int
    layer_norm: bool
    critic_lr: float
    clip_grad_norm: float | None

    num_noises: int
    feature_dim: int
    feature_lr: float
    feature_ema: float
    embed_dim: int
    phi_hidden_dims: List[int]
    mu_hidden_dims: List[int]
    critic_hidden_dims: List[int]
    reward_hidden_dims: List[int]
    rff_dim: int
    ddpm_coef: float
    reward_coef: float
    back_critic_grad: bool
    critic_coef: float

    exploration_noise: float
    num_samples: int
    temp: float
    ld: LDConfig
