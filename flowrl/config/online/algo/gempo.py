from dataclasses import dataclass

from .dpmd import DPMDConfig


@dataclass
class GeMPOConfig(DPMDConfig):
    reweight: str
    negative_bound: float
