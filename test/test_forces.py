from unittest import TestCase, main
from phys import Gravity, Electromagnetism
from phys import Particle
import numpy as np

class TestForces(TestCase):
    def test_gravity (self):
        engine = Gravity(G=1.0)
        p1 = Particle(
            mass=1.0,
            position=np.array([0.5, 0, 0]),
            velocity=np.array([0, 0.5, 0]),
            charge=1.0,
        )
        p2 = Particle(
            mass=1.0,
            position=np.array([-0.5, 0, 0]),
            velocity=np.array([0, -0.5, 0]),
            charge=1.0,
        )
        engine_force = np.linalg.norm(engine.interact(p1, p2))
        true_force = 1.0 * p1.mass * p2.mass

        self.assertAlmostEqual(engine_force, true_force, 5)

    def test_electromagnetism (self):
        engine = Electromagnetism(k=1.0)
        p1 = Particle(
            mass=1.0,
            position=np.array([0.5, 0, 0]),
            velocity=np.array([0, 0.5, 0]),
            charge=1.0,
        )
        p2 = Particle(
            mass=1.0,
            position=np.array([-0.5, 0, 0]),
            velocity=np.array([0, -0.5, 0]),
            charge=-1.0,
        )
        engine_force_vec = engine.interact(p1, p2)
        engine_force_mag = np.linalg.norm(engine_force_vec)

        self.assertEqual(list(engine_force_vec / engine_force_mag)[0], -1.0)
        self.assertAlmostEqual(engine_force_mag, 1.0, 5)


if __name__ == "__main__":
    main()
