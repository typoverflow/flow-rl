from dataclasses import dataclass
from typing import List

from .base import BaseAlgoConfig


@dataclass
class CRTL_TD3_Config(BaseAlgoConfig):
    name: str
    discount: float
    ema: float
    actor_hidden_dims: List[int]
    critic_hidden_dims: List[int]
    phi_hidden_dims: List[int]
    mu_hidden_dims: List[int]
    reward_hidden_dims: List[int]
    activation: str
    critic_ensemble_size: int
    layer_norm: bool
    feature_lr: float
    actor_lr: float
    critic_lr: float
    clip_grad_norm: float | None
    actor_update_freq: int
    target_update_freq: int
    target_policy_noise: float
    noise_clip: float
    exploration_noise: float

    ctrl_coef: float
    critic_coef: float
    feature_tau: float
    feature_dim: int
    rff_dim: int

    aug_batch_size: int
    num_noises: int
    ranking: bool
    linear: bool
