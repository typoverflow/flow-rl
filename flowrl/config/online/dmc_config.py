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
    # base
    seed: int
    device: str
    task: str
    algo: Any
    frame_skip: int
    frame_stack: int
    batch_size: int
    buffer_size: int
    lap_alpha: float
    norm_obs: bool
    norm_reward: bool
    discount: float

    train_frames: int
    random_frames: int
    warmup_frames: int
    eval_frames: int
    lap_reset_frames: int
    log_frames: int

    log: LogConfig
    eval: EvalConfig
