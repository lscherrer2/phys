from phys.buffer import Buffer
from numpy.typing import NDArray

__all__ = ["Particle"]

class Particle(Buffer):
    num_particles: int = 0
    def __init__ (
        self,
        mass: float,
        position: NDArray,
        velocity: NDArray,
        charge: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num = Particle.num_particles
        Particle.num_particles += 1
        self.mass = mass
        self.position = position.copy().astype(float)
        self.velocity = velocity.copy().astype(float)
        self.charge = charge


