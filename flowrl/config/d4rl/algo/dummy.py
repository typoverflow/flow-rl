from .base import BaseAlgoConfig
from dataclasses import dataclass

@dataclass
class DummyConfig(BaseAlgoConfig):
    """Configuration class for the Dummy algorithm."""
    name: str = "dummy"

    discount: float = 0.99
