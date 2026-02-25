from dataclasses import dataclass
from typing import List, Optional

from .base import BaseAlgoConfig


@dataclass
class FPOConfig(BaseAlgoConfig):
    """Flow Policy Optimization (FPO) configuration.

    FPO uses rectified-flow / linear-interpolation flow matching with PPO-style clipping.
    Paper: https://arxiv.org/abs/2507.21053

    Key FPO-specific settings:
    - ratio = exp(mean(L_old) - mean(L_new)) where L is CFM loss
    - Continuous t sampling (matches CNF.linear_interpolation)
    - Shared (eps, t) samples between old and new loss evaluation
    """
    name: str
    backbone_cls: str  # "mlp" or "simba"

    # === Flow-specific parameters ===
    flow_hidden_dims: List[int]       # Velocity predictor architecture
    flow_steps: int                   # ODE integration steps for sampling
    n_mc: int                         # Monte Carlo samples for CFM loss estimation
    timestep_embed_dim: int           # Positional embedding dimension for time
    policy_output_scale: float        # Scale velocity output (important for stability)
    log_ratio_clip: float             # Clip log-ratio before exp to prevent overflow
    output_mode: str                  # "u" (velocity MSE) or "u_but_supervise_as_eps" (recommended)
    average_losses_before_exp: bool   # True: exp(mean(L_old)-mean(L_new)), False: mean(exp(clip(L_old-L_new)))

    # === Critic parameters ===
    critic_hidden_dims: List[int]
    critic_lr: float

    # === Training hyperparameters ===
    activation: str                   # "relu", "elu", "silu"
    actor_lr: float
    gamma: float                      # Discount factor
    gae_lambda: float                 # GAE lambda
    clip_epsilon: float               # PPO clipping (paper uses 0.05 for FPO)
    reward_scaling: float
    normalize_advantage: bool
    normalize_observations: bool
    value_loss_coeff: float           # Scale value loss relative to policy loss

    # === Rollout settings ===
    num_envs: int
    rollout_length: int
    num_minibatches: int
    num_epochs: int
    batch_size: int
    clip_grad_norm: Optional[float]
