from unittest import TestCase, main
from phys import GravityEngine, GravitationalInteraction
from phys import Particle
import numpy as np

class TestGravity(TestCase):
    def test_gravitational_force (self):
        engine = GravityEngine(G=1.0)
        p1 = Particle(
            mass=1.0,
            position=np.array([0.5, 0, 0]),
            velocity=np.array([0, 0.5, 0]),
            interactions=[GravitationalInteraction()],
        )
        p2 = Particle(
            mass=1.0,
            position=np.array([-0.5, 0, 0]),
            velocity=np.array([0, -0.5, 0]),
            interactions=[GravitationalInteraction()],
        )
        engine_force = np.linalg.norm(engine.interact(p1, p2))
        true_force = 1.0 * p1.mass * p2.mass

        self.assertAlmostEqual(engine_force, true_force, 5)












if __name__ == "__main__":
    main()
