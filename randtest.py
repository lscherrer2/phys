import astropy.units as u
import numpy as np
from phys.forces.gravity import Gravity
from phys.particle import Particle
from phys.integrators.euler import Euler

arr = np.array([1, 2, 3]) << u.km
norm = np.linalg.norm(arr)

particles =[ Particle.random() for _ in range(10) ]
G1 = Gravity(1.0)
G2 = Gravity(3.0)

Euler.integrate([G1, G2], particles, 1.0 << u.s)

