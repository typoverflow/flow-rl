from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class EvalConfig:
    interval: int
    stats_interval: int
    num_episodes: int
    num_samples: int
    temperature: Optional[float] # None is uniform, 0.0 is greedy

@dataclass
class DataConfig:
    dataset: str
    norm_obs: bool
    norm_reward: str
    batch_size: int # mini-batch size, used in pretraining and offline RL
    scan: bool # Scanning or random batch sampling of the dataset.
    clip_eps: float # Clip the action to [-(1-clip_eps), 1-clip_eps]

@dataclass
class LogConfig:
    dir: str
    tag: str # the name of the experiment, use for logging
    interval: int
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
    task: str # the environment to train on
    algo: Any
    pretrain_steps: int
    train_steps: int
    load: Optional[str]

    log: LogConfig
    data: DataConfig
    eval: EvalConfig
