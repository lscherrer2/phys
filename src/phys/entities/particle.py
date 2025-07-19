from phys.buffer import Buffer
from numpy.typing import NDArray

__all__ = ["Particle"]

class Particle(Buffer):
    def __init__ (
        self,
        mass: float,
        position: NDArray,
        velocity: NDArray,
        charge: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mass = mass
        self.position = position.copy().astype(float)
        self.velocity = velocity.copy().astype(float)
        self.charge = charge

