from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import MISSING

@dataclass
class BaseAlgoConfig:
    """Base configuration class for all algorithms."""
    name: str = MISSING
