from dataclasses import dataclass

from .fpo import FPOConfig, FPOFlowConfig

FPOPPFlowConfig = FPOFlowConfig


@dataclass
class FPOPPConfig(FPOConfig):
    cfm_loss_clamp: float = -1.0
    cfm_loss_clamp_negative_advantages_max: float = 20.0
    cfm_diff_clamp_max: float = 10.0
    advantage_clamp: float = 100.0
