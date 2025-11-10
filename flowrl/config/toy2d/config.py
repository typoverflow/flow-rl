from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DataConfig:
    batch_size: int # mini-batch size, used in pretraining and offline RL
    scan: bool # Scanning or random batch sampling of the dataset.

@dataclass
class EvalConfig:
    interval: int # Interval for plotting

@dataclass
class LogConfig:
    dir: str
    tag: str # the name of the experiment, use for logging
    interval: int
    save_interval: int
    stats_interval: int
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
    pretrain_only: bool
    pretrain_steps: int
    train_steps: int
    load: Optional[str]

    log: LogConfig
    data: DataConfig
    eval: EvalConfig
