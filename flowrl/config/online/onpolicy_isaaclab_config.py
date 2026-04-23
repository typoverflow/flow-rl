from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class EvalConfig:
    num_episodes: int
    num_samples: int


@dataclass
class LogConfig:
    dir: str
    tag: str
    save_ckpt: bool
    save_video: bool
    # wandb
    project: str
    entity: str

@dataclass
class Config:
    seed: int
    device: str
    task: str
    algo: Any
    action_bound: Optional[float]
    disable_bootstrap: bool
    norm_obs: bool
    train_frames: int
    eval_frames: int
    log_frames: int
    num_envs: int
    rollout_length: int
    log: LogConfig
    eval: EvalConfig
