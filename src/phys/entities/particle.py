from phys.buffer import Buffer
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from phys.forces import Interaction
    from numpy.typing import NDArray

__all__ = ["Particle"]

class Particle(Buffer):
    def __init__ (
        self,
        mass: float,
        position: NDArray,
        velocity: NDArray,
        interactions: Iterable[Interaction],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mass = mass
        self.position = position.copy().astype(float)
        self.velocity = velocity.copy().astype(float)
        self.interactions: dict[type[Interaction], Interaction] = {
            type(interaction): interaction for interaction in interactions
        }

