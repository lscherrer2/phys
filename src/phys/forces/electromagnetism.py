from phys.forces.engine import Engine
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from phys.entities.particle import Particle

class Electromagnetism (Engine):
    def __init__ (self, k: float, **kwargs):
        self.k = k

    def interact (self, particle: Particle, effector: Particle) -> NDArray:
        if particle is effector: return np.array([0, 0, 0], dtype=float)

        r_vec = effector.position - particle.position
        r_mag = np.linalg.norm(r_vec)
        if r_mag == 0: return np.array([0, 0, 0], dtype=float)

        r_dir = r_vec / r_mag

        f_mag = self.k * particle.charge * effector.charge / (r_mag ** 2)
        f_vec = r_dir * f_mag

        return f_vec

