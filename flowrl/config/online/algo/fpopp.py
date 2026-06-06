from dataclasses import dataclass
from typing import List, Optional

from .base import BaseAlgoConfig


@dataclass
class FPOPPFlowConfig:
    activation: str
    hidden_dims: List[int]
    time_dim: int
    steps: int
    clip_sampler: bool
    lr: float
    x_min: float
    x_max: float


@dataclass
class FPOPPConfig(BaseAlgoConfig):
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
    flow: FPOPPFlowConfig
    # CFM loss / ratio handling
    cfm_loss_reduction: str = "sqrt"
    cfm_loss_t_inverse_cdf_beta: float = 1.0
    cfm_loss_clamp: float = 20.0
    cfm_loss_clamp_negative_advantages_max: float = 20.0
    cfm_diff_clamp_max: float = 10.0
    advantage_clamp: float = 100.0
    # Value loss
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = False
    # Action handling (clip_sampler=false; clip final action in the algorithm)
    action_clip: float = 2.0
    # EMA of the actor (flow) weights, used for deterministic/eval sampling
    ema_decay: float = 0.95
    ema_warmup_steps: int = 500
