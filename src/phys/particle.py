from __future__ import annotations
import astropy.units as u
from phys.buffer import Buffer
import numpy as np

class Particle(Buffer):

    __slots__ = ('mass', 'charge', 'position', 'velocity', 'id', 'buffer')
    spawned: int = 0

    def __init__ (
        self,
        mass: u.Quantity,
        charge: u.Quantity,
        position: u.Quantity,
        velocity: u.Quantity,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.id = Particle.spawned
        Particle.spawned += 1

        self.mass: u.Quantity = mass
        self.charge: u.Quantity = charge
        self.position: u.Quantity = position
        self.velocity: u.Quantity = velocity

    @classmethod
    def random (cls) -> Particle:
        return cls(
            mass=np.random.randn() << u.kg,
            charge=np.random.randn() << u.C,
            position=np.random.randn(3,) << u.m,
            velocity=np.random.randn(3,) << u.m / u.s,
        )

