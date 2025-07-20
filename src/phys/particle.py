import astropy.units as u
from phys.buffer import Buffer

class Particle(Buffer):
    def __init__ (
        self,
        mass: u.Quantity,
        charge: u.Quantity,
        position: u.Quantity,
        velocity: u.Quantity,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mass = mass
        self.charge = charge
        self.position = position
        self.velocity = velocity
