from unittest import TestCase, main
from phys.forces.gravity import Gravity
from phys.particle import Particle
import numpy as np
import numpy.testing as npt
import astropy.units as u

class TestForces(TestCase):
    def test_gravity (self):
        engine = Gravity(G=1.0)
        p1 = Particle(
            mass=1.0 << u.kg,
            position=np.array([0.5, 0, 0]) << u.m,
            velocity=np.array([0, 0.5, 0]) << u.m,
            charge=1.0 << u.C,
        )
        p2 = Particle(
            mass=1.0 << u.kg,
            position=np.array([-0.5, 0, 0]) << u.m,
            velocity=np.array([0, -0.5, 0]) << u.m,
            charge=1.0 << u.C,
        )
        forces = engine.interact([p1, p2])
        true_forces = np.array(
            [[-1.0, 0.0, 0.0],
             [ 1.0, 0.0, 0.0]]
        ) << u.N
        # g m1 m2 / r**2

        npt.assert_almost_equal(true_forces.value, forces.value, 5)



if __name__ == "__main__":
    main()
