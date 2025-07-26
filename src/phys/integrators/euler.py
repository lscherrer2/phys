from phys.integrators.integrator import Integrator
from phys.particle import Particle
from phys.forces.engine import Engine
import numpy as np
import astropy.units as u

__all__ = ["Euler"]

class Euler (Integrator):

    @staticmethod
    def integrate (engines: list[Engine], particles: list[Particle], timestep: u.Quantity):
        forces = Integrator.forces(engines, particles)
        masses = u.Quantity([particle.mass for particle in particles])
        # Reshape masses to (N, 1) for proper broadcasting with forces (N, 3)
        masses = masses.reshape(-1, 1)
        accels = forces / masses
        delta_vs = accels * timestep

        for i, particle in enumerate(particles):
            particle.buffer["position"] = particle.position + particle.velocity * timestep
            particle.buffer["velocity"] = particle.velocity + delta_vs[i]
