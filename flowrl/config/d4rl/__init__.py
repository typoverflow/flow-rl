from hydra.core.config_store import ConfigStore
from .config import Config, LogConfig, DataConfig, EvalConfig
from .algo.base import BaseAlgoConfig
from .algo.dummy import DummyConfig

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

cs.store(group="algo", name="dummy", node=DummyConfig)