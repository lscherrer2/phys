from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from phys.entities.particle import Particle

__all__ = ["Engine"]

class Engine (ABC):

    def __init__ (self, **kwargs):
        pass

    @abstractmethod
    def interact (self, particles: Particle, effectors: list[Particle]) -> list[NDArray]:
        """Calculates the force between particles"""
        ...


