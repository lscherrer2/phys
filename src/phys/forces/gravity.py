from phys.forces.engine import Engine
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from phys.entities.particle import Particle

class Gravity (Engine):
    def __init__ (self, G = 6.67430e-11, **kwargs):
        super().__init__(**kwargs)
        self.G = G

    def interact (self, particle: Particle, effector: Particle) -> NDArray:
        if particle is effector: return np.array([0, 0, 0], dtype=float)

        r_vec = effector.position - particle.position
        r_mag = np.linalg.norm(r_vec)
        if r_mag == 0: return np.array([0, 0, 0], dtype=float)

        r_dir = r_vec / r_mag

        f_mag = self.G * particle.mass * effector.mass / (r_mag ** 2)
        f_vec = r_dir * f_mag

        return f_vec
