from dataclasses import dataclass
from typing import List, Optional

from .base import BaseAlgoConfig


@dataclass
class GenPOFlowConfig:
    activation: str
    hidden_dims: List[int]
    time_dim: int
    steps: int
    mix_para: float
    lr: float


@dataclass
class GenPOConfig(BaseAlgoConfig):
    name: str
    backbone_cls: str
    critic_hidden_dims: List[int]
    critic_activation: str
    critic_lr: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    reward_scaling: float
    normalize_advantage: bool
    num_envs: int
    rollout_length: int
    num_minibatches: int
    num_epochs: int
    batch_size: int
    clip_grad_norm: Optional[float]
    entropy_coeff: float
    compress_coef: float
    flow: GenPOFlowConfig
