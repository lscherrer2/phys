import numpy as np
from numpy.typing import NDArray, SupportsFloat

__all__ = ["Object"]

class Object:
    def __init__ (
        self, 
        mass: float = 1, 
        pos: NDArray | list[SupportsFloat] = np.array([0.0, 0.0, 0.0]), 
        vel: NDArray | list[SupportsFloat] = np.array([0.0, 0.0, 0.0]),

    ):
        self.mass: float = mass
        self.pos: NDArray = np.asarray(pos, dtype=float)
        self.vel: NDArray = np.asarray(vel, dtype=float)

    @staticmethod
    def distance (a: "Object", b: "Object"):
        return np.sqrt((a.pos - b.pos)**2)

