from hydra.core.config_store import ConfigStore

from .algo.alac import ALACConfig
from .algo.base import BaseAlgoConfig
from .algo.ctrl import *
from .algo.dpmd import DPMDConfig
from .algo.idem import IDEMConfig
from .algo.qsm import QSMConfig
from .algo.sac import SACConfig
from .algo.sdac import SDACConfig
from .algo.td3 import TD3Config
from .algo.td7 import TD7Config
from .algo.unirep import *
from .config import Config, LogConfig

_DEF_SUFFIX = "_cfg_def"

cs = ConfigStore.instance()
cs.store(name="config" + _DEF_SUFFIX, node=Config)

# raise error if algo is not specified
cs.store(group="algo", name="base", node=BaseAlgoConfig)

_CONFIGS = {
    "sac": SACConfig,
    "sdac": SDACConfig,
    "td3": TD3Config,
    "td7": TD7Config,
    "dpmd": DPMDConfig,
    "qsm": QSMConfig,
    "alac": ALACConfig,
    "idem": IDEMConfig,
    "ctrl_td3": CtrlTD3Config,
    "ctrl_qsm": CtrlQSMConfig,
    "aca": ACAConfig,
}

for name, cfg in _CONFIGS.items():
    cs.store(group="algo", name=name + _DEF_SUFFIX, node=cfg)
