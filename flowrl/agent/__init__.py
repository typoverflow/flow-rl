from typing import Dict, Type
from .base import BaseAgent

from .dummy import DummyAgent
from .iql import IQLAgent


SUPPORTED_AGENTS: Dict[str, Type[BaseAgent]] = {
    "dummy": DummyAgent,
    "iql": IQLAgent,
    # "bdpo_discrete": BDPO_Discrete, 
    # "DAC": DAC, 
}