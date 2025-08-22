from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class BaseAlgoConfig:
    """Base configuration class for all algorithms."""
    name: str
