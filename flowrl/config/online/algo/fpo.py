from dataclasses import dataclass
from typing import List, Optional

from .base import BaseAlgoConfig


@dataclass
class FPOFlowConfig:
    activation: str
    hidden_dims: List[int]
    time_dim: int
    steps: int
    clip_sampler: bool
    lr: float
    x_min: float
    x_max: float
    output_scale: float


@dataclass
class FPOConfig(BaseAlgoConfig):
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
    additive_noise: float
    num_mc_samples: int
    flow: FPOFlowConfig
