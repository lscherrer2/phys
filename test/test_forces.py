from unittest import TestCase, main
from phys.forces.gravity import Gravity
from phys.particle import Particle
import numpy as np
import numpy.testing as npt
from astropy.units import Quantity, kg, m, C, N


class TestForces(TestCase):
    def test_gravity(self):
        engine = Gravity(G=2.0)
        p1 = Particle(
            mass=1.0 << kg,
            position=np.array([0.5, 0, 0]) << m,
            velocity=np.array([0, 0.5, 0]) << m,
            charge=1.0 << C,
        )
        p2 = Particle(
            mass=1.0 << kg,
            position=np.array([-0.5, 0, 0]) << m,
            velocity=np.array([0, -0.5, 0]) << m,
            charge=1.0 << C,
        )
        forces = engine.interact([p1, p2])

        # g m1 m2 / r**2
        true_forces: Quantity = np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]) << N

        npt.assert_almost_equal(forces.value, true_forces.value, 5)


if __name__ == "__main__":
    main()
