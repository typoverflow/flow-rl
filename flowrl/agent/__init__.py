from typing import Dict, Type

from .base import BaseAgent
from .bdpo.bdpo import BDPOAgent
from .iql import IQLAgent
from .ivr import IVRAgent

__all__ = [
    "BaseAgent",
    "IQLAgent",
    "BDPOAgent",
    "IVRAgent"
]
