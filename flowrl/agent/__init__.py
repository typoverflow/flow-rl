from typing import Dict, Type

from .base import BaseAgent
from .dummy import DummyAgent
from .iql import IQLAgent

__all__ = [
    "BaseAgent",
    "DummyAgent",
    "IQLAgent",
]
