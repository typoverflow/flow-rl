from hydra.core.config_store import ConfigStore

from .algo.base import BaseAlgoConfig
from .algo.dpmd import DPMDConfig
from .algo.sac import SACConfig
from .algo.sdac import SDACConfig
from .algo.td3 import TD3Config
from .algo.td7 import TD7Config
from .config import Config, LogConfig

_DEF_SUFFIX = "_cfg_def"

cs = ConfigStore.instance()
cs.store(name="config"+_DEF_SUFFIX, node=Config)

# raise error if algo is not specified
cs.store(group="algo", name="base", node=BaseAlgoConfig)

_CONFIGS = {
    "sac": SACConfig,
    "sdac": SDACConfig,
    "td3": TD3Config,
    "td7": TD7Config,
    "dpmd": DPMDConfig,
}

for name, cfg in _CONFIGS.items():
    cs.store(group="algo", name=name+_DEF_SUFFIX, node=cfg)
