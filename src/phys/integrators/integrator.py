from abc import ABC, abstractmethod
from phys.particle import Particle
from phys.forces.engine import Engine
import astropy.units as u
import numpy as np

class Integrator (ABC):
    @staticmethod
    def forces (engines: list[Engine], particles: list[Particle]):
        engine_forces = np.concat([engine.interact(particles) for engine in engines], axis=1)
        # dims: engine, particle, coord
        net_forces = np.sum(engine_forces, axis=0)
        return net_forces

    @staticmethod
    @abstractmethod
    def integrate (engines: list[Engine], particles: list[Particle], timestep: u.Quantity):
       pass
