from .base import BaseAlgoConfig
from dataclasses import dataclass, field
from typing import Any, List, Optional

@dataclass
class DummyConfig(BaseAlgoConfig):
    """Configuration class for the Dummy algorithm."""
    name: str = "dummy"

    discount: float = 0.99
