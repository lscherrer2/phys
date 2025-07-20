from unittest import TestCase, main
from phys.forces.gravity import Gravity
from phys.particle import Particle
import astropy.units as u
import numpy as np

class TestEngine (TestCase):
    def test_interact (self):
        engine = Gravity(1.0)
        particles = [
            Particle(
                mass=1.0 << u.kg,
                position=np.random.randn(3) << u.m,
                velocity=np.random.randn(3) << (u.m / u.s), # type: ignore
                charge=0.0 * u.C,
            ) for _ in range(10)
        ]
        force = engine.interact(particles)
        self.assertIsInstance(force, np.ndarray)
        self.assertEqual(force.shape, (10,))

    def test_batch_interact_single_core (self):
        engine = Gravity(1.0)
        num_effectors = 10
        particles = [
            Particle(
                mass=1.0 << u.kg,
                position=np.random.randn(3) << u.m,
                velocity=np.random.randn(3) << (u.m / u.s), # type: ignore
                charge=0.0 * u.C,
            ) for _ in range(num_effectors)
        ]
        forces =

if __name__ == "__main__":
    main()
