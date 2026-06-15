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

    # drifting policy: V = base_step_size * h_t * (g/s - (lam/h_t/s) m), regress a toward sg(a + V)
    temp: float                  # temperature lambda; target pi* propto exp(Q / temp); the per-env knob
    base_step_size: float        # effective drift step = base_step_size * h_t

    # annealed bandwidth h_t: bandwidth_init -> bandwidth_end over bandwidth_anneal_steps gradient steps
    bandwidth_init: float
    bandwidth_end: float
    bandwidth_anneal_steps: int

    num_train_samples: int       # drift MC samples: a^+ ~ N(a, h_t I) and a^- ~ pi (shared count)

    num_samples: int             # best-of-N action selection at eval time
    x_min: float
    x_max: float
