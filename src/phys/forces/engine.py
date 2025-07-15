from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from numpy.typing import NDArray
if TYPE_CHECKING:
    from phys.entity import Particle
    from phys.forces.interaction import Interaction

__all__ = [
    "Engine",
]


class Engine (ABC):

    interaction_type = None

    def __init__ (self, **kwargs):
        pass

    @abstractmethod
    def interact (self, particles: Particle, effectors: list[Particle]) -> list[NDArray]:
        """Calculates the force between particles"""
        ...
