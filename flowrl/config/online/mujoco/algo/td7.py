from dataclasses import dataclass

from .base import BaseAlgoConfig


@dataclass
class TD7Config(BaseAlgoConfig):
    name: str
    discount: float
    hidden_dim: int
    embed_dim: int
    actor_lr: float
    critic_lr: float
    encoder_lr: float
    clip_grad_norm: float | None
    actor_update_freq: int
    target_update_freq: int
    target_policy_noise: float
    noise_clip: float
    exploration_noise: float
    lam: float
    max_action: float
    lap: bool
