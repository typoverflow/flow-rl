from dataclasses import dataclass
from typing import List

from .base import BaseAlgoConfig


@dataclass
class DIMESamplerConfig:
    hidden_dims: List[int]
    activation: str
    layer_norm: bool
    time_dim: int
    steps: int
    prior_std: float
    dt: float
    friction: float
    learn_dt: bool
    learn_friction: bool
    per_dim_friction: bool
    schedule_min: float
    schedule_offset: float
    schedule_power: int
    output_clip: float
    output_kernel_init: float


@dataclass
class DIMEConfig(BaseAlgoConfig):
    name: str
    critic_hidden_dims: List[int]
    critic_activation: str
    critic_ensemble_size: int
    critic_lr: float
    critic_num_atoms: int
    critic_v_min: float
    critic_v_max: float
    critic_entropy_coef: float
    batch_renorm: bool
    batch_norm_momentum: float
    batch_norm_epsilon: float
    batch_norm_warmup: int
    actor_lr: float
    actor_grad_clip: float
    adam_b1: float
    adam_b2: float
    discount: float
    actor_update_freq: int
    target_actor_ema: float
    init_temperature: float
    temperature_lr: float
    target_entropy_scale: float
    sampler: DIMESamplerConfig
