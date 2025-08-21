from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING

from .base import BaseAlgoConfig


@dataclass
class DTQLConfig(BaseAlgoConfig):
    name: str

    discount: float
    gamma: float # weight of gamma loss
    ema: float # soft target update
    expectile: float
    alpha: float # weight of distillation loss
    sigma_max: float
    sigma_min: float
    sigma_data: float

    lr: float
    lr_decay: bool
    lr_decay_steps: int

    max_action: float
    min_action: float
