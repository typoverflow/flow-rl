from dataclasses import dataclass
from typing import Any

from .hb_config import EvalConfig, LogConfig


@dataclass
class Config:
    seed: int
    device: str
    task: str
    algo: Any
    frame_skip: int
    frame_stack: int
    norm_obs: bool
    store_action_chains: bool
    train_frames: int
    eval_frames: int
    log_frames: int
    num_envs: int
    rollout_length: int
    log: LogConfig
    eval: EvalConfig
