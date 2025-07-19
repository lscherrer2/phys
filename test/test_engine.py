from unittest import TestCase, main
from phys import Gravity, Electromagnetism
from phys import Particle
import numpy as np

class TestEngines(TestCase):
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
        engine_force = np.linalg.norm(engine.interact(p1, p2))
        true_force = 1.0 * p1.charge * p2.charge

        self.assertAlmostEqual(engine_force, true_force, 5)


if __name__ == "__main__":
    main()
