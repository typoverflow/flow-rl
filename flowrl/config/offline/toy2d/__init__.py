from hydra.core.config_store import ConfigStore

from ..d4rl.algo.base import BaseAlgoConfig
from ..d4rl.algo.bdpo import BDPOConfig
from ..d4rl.algo.dac import DACConfig
# from ..d4rl.algo.dql import DQLConfig
# from ..d4rl.algo.dtql import DTQLConfig
# from ..d4rl.algo.fql import FQLConfig
# from ..d4rl.algo.iql import IQLConfig
# from ..d4rl.algo.ivr import IVRConfig
from .config import Config

_DEF_SUFFIX = "_cfg_def"

cs = ConfigStore.instance()
cs.store(name="config"+_DEF_SUFFIX, node=Config)

# raise error if algo is not specified
cs.store(group="algo", name="base", node=BaseAlgoConfig)

_CONFIGS = {
    # "iql": IQLConfig,
    "bdpo": BDPOConfig,
    # "ivr": IVRConfig,
    "dac": DACConfig,
    # "dql": DQLConfig,
    # "fql": FQLConfig,
    # "dtql": DTQLConfig,
}

for name, cfg in _CONFIGS.items():
    cs.store(group="algo", name=name+_DEF_SUFFIX, node=cfg)
