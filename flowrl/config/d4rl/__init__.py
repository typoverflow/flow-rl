from hydra.core.config_store import ConfigStore
from .config import Config, LogConfig, DataConfig, EvalConfig
from .algo.base import BaseAlgoConfig
from .algo.dummy import DummyConfig
from .algo.iql import IQLConfig

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

# raise error if algo is not specified
cs.store(group="algo", name="base", node=BaseAlgoConfig)

cs.store(group="algo", name="dummy", node=DummyConfig)
cs.store(group="algo", name="iql", node=IQLConfig)