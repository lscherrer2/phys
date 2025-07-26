from abc import ABC, abstractmethod
from phys.particle import Particle
import numpy as np
import astropy.units as u

class Engine (ABC):

    @abstractmethod
    def force (self, particle: Particle, effector: Particle) -> u.Quantity:
        pass

    def interact (self, particles: list[Particle]) -> u.Quantity:
        count = len(particles)
        force_matrix = np.zeros((count, count, 3)) << u.N
        for p, particle in enumerate(particles):
            for e, effector in enumerate(particles[p+1:]):
                pe_force = self.force(particle, effector)
                force_matrix[p, e] = pe_force
                force_matrix[e, p] = -1.0 * pe_force
        net_forces = np.sum(force_matrix, axis=1)
        return net_forces


