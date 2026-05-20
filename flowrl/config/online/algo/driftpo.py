from dataclasses import dataclass
from typing import List

from .base import BaseAlgoConfig


@dataclass
class DriftPOConfig(BaseAlgoConfig):
    name: str
    backbone_cls: str

    actor_hidden_dims: List[int]
    actor_activation: str
    actor_lr: float
    noise_dim: int

    critic_hidden_dims: List[int]
    critic_activation: str
    critic_lr: float
    critic_ensemble_size: int

    discount: float
    ema: float

    # drifting policy specific
    opt_method: str        # "first" or "zeroth"
    bandwidth: float       # kernel bandwidth h (floor when bandwidth_mode='median')
    bandwidth_mode: str    # "fixed" or "median"
    temp: float            # temperature lambda; meaning depends on temp_mode
    temp_mode: str         # "fixed" | "normalize" | "balance" | "raw"
    pos_strength: float    # for temp_mode='balance', target ratio ||V_pos|| / ||V_neg|| per sample
    drift_scale: float     # global scaling for V (step size)
    max_step: float        # if > 0, clip ||V||_2 to this radius before forming the target
    num_pos_samples: int   # MC samples for zeroth-order V^+
    num_neg_samples: int   # MC samples for V^- (kernelized mean-shift)

    num_samples: int       # best-of-N action selection at eval time
    x_min: float
    x_max: float
