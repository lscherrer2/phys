
from phys.forces.interaction import Interaction
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from phys.entity import Particle

class GravityEngine (Interaction):
    def __init__ (self, G = 6.67430e-11, **kwargs):
        super().__init__(**kwargs)
        self.G = G

    def interact (self, particle: Particle, effectors: list[Particle]) -> list[NDArray]:
        forces = []
        for effector in effectors:
            if not
                raise ValueError(
                    "Attempted to interact particle without Gravity inteaction does not "
                )
            r_vec: NDArray = particle.position.copy()
            r_mag = np.linalg.norm(r_vec)





