from phys.integrators.integrator import Integrator
from phys.particle import Particle
from phys.forces.engine import Engine
import numpy as np
import astropy.units as u

class Euler (Integrator):

    @staticmethod
    def integrate (engines: list[Engine], particles: list[Particle], timestep: u.Quantity):
        forces = Integrator.forces(engines, particles)
        masses = np.concat([particle.mass for particle in particles]).reshape(-1, 1)
        accels = forces / masses
        delta_vs = accels * timestep

        for i, particle in enumerate(particles):
            particle.buffer["position"] = particle.position + particle.velocity * timestep
            particle.buffer["velocity"] = particle.velocity + delta_vs[i]
