from typing import Dict, Type

from .base import BaseAgent
from .bdpo.bdpo import BDPOAgent
from .dac import DACAgent
from .fql.fql import FQLAgent
from .iql import IQLAgent
from .ivr import IVRAgent

__all__ = [
    "BaseAgent",
    "IQLAgent",
    "BDPOAgent",
    "IVRAgent",
    "FQLAgent",
    "DACAgent",
]
