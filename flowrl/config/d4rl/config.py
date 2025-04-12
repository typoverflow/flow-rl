from dataclasses import dataclass, field
from typing import Any, List, Optional
from .algo.base import BaseAlgoConfig
from omegaconf import MISSING


@dataclass
class EvalConfig:
    interval: int = 50000
    stats_interval: int = 2000
    num_episodes: int = 10
    num_samples: int = 10
    temperature: Optional[float] = 0.0 # None is uniform, 0.0 is greedy

@dataclass
class DataConfig:
    dataset: str = "d4rl"
    norm_reward: str = "iql_mujoco"
    batch_size: int = 256 # mini-batch size, used in pretraining and offline RL
    scan: bool = True # Scanning or random batch sampling of the dataset.
    clip_eps: float = 0.0 # Clip the action to [-(1-clip_eps), 1-clip_eps]

@dataclass
class LogConfig:
    dir: str = "logs"
    tag: str = "debug"  # the name of the experiment, use for logging
    interval: int = 500
    save_ckpt: bool = False
    save_video: bool = False

@dataclass
class Config:
    # base
    seed: int = 0
    num_seeds: int = 1 # number of runs of different seeds
    gpu: str = "0" # dot seperated string
    env: str = "hopper-medium-replay-v2"  # the environment to train on
    algo: BaseAlgoConfig = MISSING
    pretrain_steps: int = int(2e6)
    train_steps: int = int(2e6)
    norm_obs: bool = False
    mode: str = "train" # ["pretrain" or "train" or "debug"]
    load: Optional[str] = None

    log: LogConfig = field(default_factory=LogConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Hydra config
    hydra: Any = field(default_factory=lambda: {
        "output_subdir": None,
        "run": {"dir": "."},
    })
    defaults: List[Any] = field(default_factory=lambda: [
        "_self_",
        {"override hydra/hydra_logging": "disabled"},
        {"override hydra/job_logging": "disabled"},
    ])
