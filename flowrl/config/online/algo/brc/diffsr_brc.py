from dataclasses import dataclass
from typing import List

from ..base import BaseAlgoConfig


@dataclass
class DiffSRBRCConfig(BaseAlgoConfig):
    name: str
    num_updates: int
    num_bins: int
    v_max: float
    discount: float
    ema: float
    actor_hidden_dim: int
    critic_hidden_dim: int
    critic_ensemble_size: int
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    clip_grad_norm: float | None

    num_noises: int
    feature_dim: int
    feature_lr: float
    embed_dim: int
    phi_hidden_dims: List[int]
    mu_hidden_dims: List[int]
    reward_hidden_dims: List[int]
    rff_dim: int
    reward_coef: float
