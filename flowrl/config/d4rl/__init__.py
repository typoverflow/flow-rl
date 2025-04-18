from hydra.core.config_store import ConfigStore

from .algo.base import BaseAlgoConfig
from .algo.bdpo import BDPOConfig
from .algo.iql import IQLConfig
from .config import Config, DataConfig, EvalConfig, LogConfig

_DEF_SUFFIX = "_cfg_def"

cs = ConfigStore.instance()
cs.store(name="config"+_DEF_SUFFIX, node=Config)

# raise error if algo is not specified
cs.store(group="algo", name="base", node=BaseAlgoConfig)

_CONFIGS = {
    "iql": IQLConfig,
    "bdpo": BDPOConfig,
}

for name, cfg in _CONFIGS.items():
    cs.store(group="algo", name=name+_DEF_SUFFIX, node=cfg)
