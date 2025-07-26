from phys.forces.engine import Engine
from phys.particle import Particle
import astropy.units as u
import numpy as np

class Gravity (Engine):

    _symmetric = True

    def __init__ (self, G: u.Quantity | float = 6e-11):
        self.G = (
            G.to(u.m**3 / (u.kg * u.s**2))
            if isinstance(G, u.Quantity)
            else G * (u.m**3 / (u.kg * u.s**2))
        )

    def force (self, particle: Particle, effector: Particle) -> u.Quantity:
        r_vec: u.Quantity = effector.position - particle.position # type: ignore
        r_mag: u.Quantity = np.linalg.norm(r_vec) # type: ignore
        r_direction: u.Quantity = r_vec / r_mag

        f_mag: u.Quantity = self.G * particle.mass * effector.mass / (r_mag ** 2)
        f_mag = f_mag.to(u.N) # type: ignore

        f_vec = f_mag * r_direction
        return f_vec # type: ignore
