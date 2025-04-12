from typing import Dict, Type
from .base import BaseAgent

from .dummy import DummyAgent


SUPPORTED_AGENTS: Dict[str, Type[BaseAgent]] = {
    "dummy": DummyAgent, 
    # "bdpo_discrete": BDPO_Discrete, 
    # "DAC": DAC, 
}