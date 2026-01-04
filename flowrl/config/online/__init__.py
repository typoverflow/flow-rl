from hydra.core.config_store import ConfigStore

from .algo.base import BaseAlgoConfig
from .algo.ctrl_td3 import CtrlTD3Config
from .algo.dpmd import DPMDConfig
from .algo.sac import SACConfig
from .algo.sdac import SDACConfig
from .algo.td3 import TD3Config
from .algo.td7 import TD7Config
from .dmc_config import Config as DMCCONFIG
from .hb_config import Config as HBConfig
from .mujoco_config import Config as MUJOCOConfig

_DEF_SUFFIX = "_cfg_def"

cs = ConfigStore.instance()
cs.store(name="mujoco_config" + _DEF_SUFFIX, node=MUJOCOConfig)
cs.store(name="hb_config" + _DEF_SUFFIX, node=HBConfig)
cs.store(name="dmc_config" + _DEF_SUFFIX, node=DMCCONFIG)

# raise error if algo is not specified
cs.store(group="algo", name="base", node=BaseAlgoConfig)

_CONFIGS = {
    "sac": SACConfig,
    "sdac": SDACConfig,
    "td3": TD3Config,
    "td7": TD7Config,
    "dpmd": DPMDConfig,
    "ctrl": CtrlTD3Config,
}

for name, cfg in _CONFIGS.items():
    cs.store(group="algo", name=name + _DEF_SUFFIX, node=cfg)
