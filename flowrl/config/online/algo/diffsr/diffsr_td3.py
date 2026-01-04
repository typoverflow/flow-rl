from dataclasses import dataclass
from typing import List

from ..base import BaseAlgoConfig


@dataclass
class DiffSRTD3Config(BaseAlgoConfig):
    name: str
    actor_update_freq: int
    target_update_freq: int
    discount: float
    ema: float
    actor_hidden_dims: List[int]
    # critic_hidden_dims: List[int]
    critic_ensemble_size: int
    layer_norm: bool
    actor_lr: float
    critic_lr: float
    clip_grad_norm: float | None
    target_policy_noise: float
    noise_clip: float
    exploration_noise: float

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
