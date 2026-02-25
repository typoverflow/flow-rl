from dataclasses import dataclass
from typing import Any

from .hb_config import EvalConfig, LogConfig


@dataclass
class Config:
    seed: int
    device: str
    task: str
    algo: Any
    action_bound: float
    disable_bootstrap: bool
    norm_obs: bool
    train_frames: int
    eval_frames: int
    log_frames: int
    num_envs: int
    rollout_length: int
    log: LogConfig
    eval: EvalConfig
