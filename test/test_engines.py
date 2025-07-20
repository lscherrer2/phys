from unittest import TestCase, main
from phys import Gravity, Particle, Electromagnetism
import numpy as np

class TestEngine (TestCase):
    def test_interact (self):
        engine = Gravity(1.0)
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
        force = engine.interact(p1, p2)
        self.assertIsInstance(force, np.ndarray)
        self.assertEqual(force.shape, (3,))

    def test_batch_interact_single_core (self):
        engine = Electromagnetism(1.0)
        p1 = Particle(
            mass=1.0,
            position=np.array([0.5, 0, 0]),
            velocity=np.array([0, 0.5, 0]),
            charge=1.0,
        )
        num_effectors = 10
        effectors = []
        for _ in range(num_effectors):
            particle = Particle(
                mass=np.random.randn(1).item(),
                position=np.random.randn(3),
                velocity=np.random.randn(3),
                charge=np.random.randn(1).item(),
            )
            effectors.append(particle)

        forces = engine.batch_interact(
            particle=p1,
            effectors=effectors,
            cores=1,
        )
        self.assertIsInstance(forces, dict)
        for force in forces.values():
            self.assertIsInstance(force, np.ndarray)

        self.assertEqual(len(forces), num_effectors)
        
if __name__ == "__main__":
    main()







