from dataclasses import dataclass
from typing import List, Optional

from .base import BaseAlgoConfig


@dataclass
class DPPODiffusionConfig:
    hidden_dims: List[int]
    activation: str
    time_dim: int
    steps: int
    noise_schedule: str
    clip_sampler: bool
    x_min: float
    x_max: float
    solver: str
    min_sampling_denoising_std: float
    min_logprob_denoising_std: float


@dataclass
class DPPOConfig(BaseAlgoConfig):
    name: str
    backbone_cls: str
    critic_hidden_dims: List[int]
    critic_activation: str
    actor_lr: float
    critic_lr: float
    gamma: float
    gae_lambda: float
    gamma_denoising: float
    clip_epsilon: float
    clip_epsilon_base: float
    clip_epsilon_rate: float
    reward_scaling: float
    normalize_advantage: bool
    num_envs: int
    rollout_length: int
    num_minibatches: int
    num_epochs: int
    batch_size: int
    clip_grad_norm: Optional[float]
    diffusion: DPPODiffusionConfig
