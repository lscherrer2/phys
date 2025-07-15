
from phys.forces.interaction import Interaction
from phys.forces.gravity.interaction import GravitationalInteraction
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from phys.entities.particle import Particle

class GravityEngine (Interaction):
    def __init__ (self, G = 6.67430e-11, **kwargs):
        super().__init__(**kwargs)
        self.G = G

    def interact (self, particle: Particle, effector: Particle) -> NDArray:

        # Particle don't interact with themselves or particles without Gravity
        if effector is particle or GravitationalInteraction not in effector.interactions:
            return np.array([0, 0, 0], dtype=float)

        interaction: GravitationalInteraction = effector.interactions[GravitationalInteraction] # type: ignore

        r_vec = effector.position - particle.position
        r_mag = np.linalg.norm(r_vec)

        # f = Gm1m2/r^2
        f_mag = self.G * particle.mass * effector.mass / r_mag**2

        # f_vec = f * direction
        f_vec = f_mag * (r_vec / r_mag)

        return f_vec






