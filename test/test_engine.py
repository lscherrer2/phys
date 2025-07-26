from unittest import TestCase, main
from phys import Particle, Engine
from astropy.units import Quantity, kg, m, s, C, N
import numpy as np
import numpy.testing as npt


class DistanceForceEngine(Engine):
    symmetric = False

    def force(self, particle: Particle, effector: Particle) -> Quantity:
        distance: Quantity = np.linalg.norm(particle.position - effector.position).to(m)
        return [distance.value, 0.0, 0.0] << N


class TestEngine(TestCase):
    def setUp(self):
        self.engine = DistanceForceEngine()

    def test_force(self):
        p1 = Particle(
            mass=1.0 << kg,
            charge=1.0 << C,
            position=[0.0, 0.0, 0.0] << m,
            velocity=[0.0, 0.0, 0.0] << m / s,
        )
        p2 = Particle(
            mass=1.0 << kg,
            charge=1.0 << C,
            position=[1.0, 0.0, 0.0] << m,
            velocity=[0.0, 0.0, 0.0] << m / s,
        )
        force = self.engine.force(p1, p2)
        self.assertEqual(force.shape, (3,))
        npt.assert_equal(force.value, np.array([1.0, 0.0, 0.0]))

    def test_interact(self):
        particles = [
            Particle(
                mass=1.0 << kg,
                position=[float(i), 0.0, 0.0] << m,
                velocity=[0.0, 0.0, 0.0] << (m / s),
                charge=0.0 * C,
            )
            for i in range(10)
        ]
        force = self.engine.interact(particles)
        self.assertIsInstance(force, np.ndarray)
        self.assertEqual(force.shape, (10, 3))

        expected_forces = np.zeros((10, 3))
        for i in range(10):
            net_force_x = 0.0
            for j in range(10):
                if i != j:
                    distance = abs(i - j)
                    net_force_x += distance
            expected_forces[i] = [net_force_x, 0.0, 0.0]

        npt.assert_array_almost_equal(force.value, expected_forces, decimal=10)
        self.assertEqual(force.unit, N)
        self.assertAlmostEqual(force[0, 0].value, 45.0)
        self.assertAlmostEqual(force[1, 0].value, 37.0)
        self.assertAlmostEqual(force[5, 0].value, 25.0)
        npt.assert_array_equal(force[:, 1].value, np.zeros(10))
        npt.assert_array_equal(force[:, 2].value, np.zeros(10))


if __name__ == "__main__":
    main()
