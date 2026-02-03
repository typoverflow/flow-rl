from dataclasses import dataclass
from typing import List

from .base import BaseAlgoConfig
from .qsm import QSMConfig, QSMDiffusionConfig

IDEMDiffusionConfig = QSMDiffusionConfig


@dataclass
class IDEMConfig(QSMConfig):
    name: str
    critic_hidden_dims: List[int]
    critic_activation: str
    critic_lr: float
    discount: float
    num_samples: int
    num_reverse_samples: int
    ema: float
    temp: float
    diffusion: IDEMDiffusionConfig
