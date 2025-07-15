from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from numpy.typing import NDArray
if TYPE_CHECKING:
    from phys.entity import Particle

__all__ = [
    "Engine",
]


class Engine (ABC):
    def __init__ (self, **kwargs):
        pass

    @abstractmethod
    def interact (self, particles: list[Particle], effectors: list[Particle]) -> list[NDArray]: 
        """Calculates the force between two particles"""
        ...
