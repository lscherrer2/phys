from abc import ABC, abstractmethod
from phys.particle import Particle
import numpy as np
import astropy.units as u

class Engine (ABC):

    @abstractmethod
    def force (self, particle: Particle, effector: Particle) -> u.Quantity:
        pass

    def interact (self, particles: list[Particle]):
        num_particles = len(particles)
        force_matrix = np.zeros((num_particles, num_particles, 3), dtype=float) * u.N
        for p, particle in enumerate(particles):
            for e, effector in enumerate(particles[p+1:]):
                force = self.force(particle, effector)
                force_matrix[p, e] = force
                force_matrix[e, p] = -force
        net_forces = np.sum(force_matrix, axis=1)
        return net_forces.reshape(-1)
