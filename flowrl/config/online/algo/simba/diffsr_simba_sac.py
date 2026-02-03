from dataclasses import dataclass
from typing import List

from ..base import BaseAlgoConfig


@dataclass
class DiffSRSimbaSACConfig(BaseAlgoConfig):
    name: str
    actor_num_blocks: int
    actor_hidden_dim: int
    actor_lr: float
    actor_wd: float

    critic_num_blocks: int
    critic_hidden_dim: int
    critic_lr: float
    critic_wd: float
    critic_ensemble_size: int

    alpha_lr: float
    alpha_wd: float
    alpha_init_value: float
    target_entropy_scale: float

    ema: float
    discount: float
    num_updates: int

    num_noises: int
    feature_dim: int
    feature_lr: float
    embed_dim: int
    phi_hidden_dims: List[int]
    mu_hidden_dims: List[int]
    reward_hidden_dims: List[int]
    rff_dim: int
    reward_coef: float
    clip_grad_norm: float | None
