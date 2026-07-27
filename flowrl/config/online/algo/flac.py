from dataclasses import dataclass
from typing import List

from .base import BaseAlgoConfig


@dataclass
class FLACFlowConfig:
    hidden_dims: List[int]
    activation: str
    layer_norm: bool
    initializer: str
    lr: float
    steps: int


@dataclass
class FLACConfig(BaseAlgoConfig):
    name: str
    critic_backbone_cls: str
    critic_hidden_dims: List[int]
    critic_activation: str
    critic_layer_norm: bool
    critic_initializer: str
    critic_ensemble_size: int
    critic_lr: float
    distributional_critic: bool
    critic_num_atoms: int
    critic_v_min: float
    critic_v_max: float
    discount: float
    ema: float
    reward_scale: float
    update_actor_every: int
    target_kinetic_coef: float
    init_log_alpha: float
    alpha_lr: float
    clip_grad_norm: float | None
    flow: FLACFlowConfig
