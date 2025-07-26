from unittest import TestCase, main
from phys.integrators.integrator import Integrator
from phys.integrators.euler import Euler
from phys.forces.gravity import Gravity
from phys.forces.engine import Engine
from phys.particle import Particle
import astropy.units as u
import numpy as np
import numpy.testing as npt


class MockSymmetricEngine(Engine):
    """A mock engine that applies constant force between particles."""

    _symmetric: bool = True

    def __init__(self, force_magnitude: float = 1.0):
        super().__init__()
        self.force_magnitude = force_magnitude

    def force(self, particle: Particle, effector: Particle) -> u.Quantity:
        return np.array([self.force_magnitude, 0.0, 0.0]) << u.N


class MockAsymmetricEngine(Engine):
    """A mock engine that applies constant force to all particles independently."""

    _symmetric: bool = False

    def __init__(self, force_magnitude: float = 1.0):
        super().__init__()
        self.force_magnitude = force_magnitude

    def force(self, particle: Particle, effector: Particle) -> u.Quantity:
        return np.array([self.force_magnitude, 0.0, 0.0]) << u.N


class TestIntegrator(TestCase):

    def setUp(self):
        """Set up test particles."""
        self.particle1 = Particle(
            mass=1.0 << u.kg,
            position=np.array([0.0, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),  # type: ignore
            charge=0.0 << u.C,
        )
        self.particle2 = Particle(
            mass=2.0 << u.kg,
            position=np.array([1.0, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 1.0, 0.0]) << (u.m / u.s),  # type: ignore
            charge=0.0 << u.C,
        )

    def test_forces_symmetric_engine(self):
        """Test forces calculation with symmetric interaction."""
        mock_engine = MockSymmetricEngine(force_magnitude=1.0)
        engines = [mock_engine]  # type: ignore
        particles = [self.particle1, self.particle2]

        forces = Integrator.forces(engines, particles)  # type: ignore

        self.assertEqual(forces.shape, (2, 3))

        expected_force1 = np.array([1.0, 0.0, 0.0]) << u.N
        expected_force2 = np.array([-1.0, 0.0, 0.0]) << u.N

        npt.assert_array_almost_equal(forces[0].value, expected_force1.value)
        npt.assert_array_almost_equal(forces[1].value, expected_force2.value)

    def test_forces_asymmetric_engine(self):
        """Test forces calculation with asymmetric interaction."""
        mock_engine = MockAsymmetricEngine(force_magnitude=1.0)
        engines = [mock_engine]  # type: ignore
        particles = [self.particle1, self.particle2]

        forces = Integrator.forces(engines, particles)  # type: ignore

        self.assertEqual(forces.shape, (2, 3))

        expected_force = np.array([1.0, 0.0, 0.0]) << u.N

        npt.assert_array_almost_equal(forces[0].value, expected_force.value)
        npt.assert_array_almost_equal(forces[1].value, expected_force.value)

    def test_forces_multiple_engines(self):
        """Test forces calculation with multiple engines."""
        engine1 = MockSymmetricEngine(force_magnitude=1.0)
        engine2 = MockSymmetricEngine(force_magnitude=0.5)
        engines = [engine1, engine2]  # type: ignore
        particles = [self.particle1, self.particle2]

        forces = Integrator.forces(engines, particles)  # type: ignore

        expected_force1 = np.array([1.5, 0.0, 0.0]) << u.N
        expected_force2 = np.array([-1.5, 0.0, 0.0]) << u.N

        npt.assert_array_almost_equal(forces[0].value, expected_force1.value)
        npt.assert_array_almost_equal(forces[1].value, expected_force2.value)

    def test_euler_integration_with_gravity(self):
        """Test Euler integration with realistic gravity forces."""
        gravity_engine = Gravity(G=1.0)
        engines = [gravity_engine]  # type: ignore

        particle1 = Particle(
            mass=1.0 << u.kg,
            position=np.array([-0.5, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),  # type: ignore
            charge=0.0 << u.C,
        )
        particle2 = Particle(
            mass=1.0 << u.kg,
            position=np.array([0.5, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),  # type: ignore
            charge=0.0 << u.C,
        )
        particles = [particle1, particle2]

        timestep = 0.1 << u.s

        initial_pos1 = particle1.position.copy()
        initial_pos2 = particle2.position.copy()

        Euler.integrate(engines, particles, timestep)  # type: ignore

        self.assertTrue("position" in particle1.buffer)
        self.assertTrue("velocity" in particle1.buffer)
        self.assertTrue("position" in particle2.buffer)
        self.assertTrue("velocity" in particle2.buffer)

        npt.assert_array_equal(particle1.position.value, initial_pos1.value)
        npt.assert_array_equal(particle2.position.value, initial_pos2.value)

        for p in particles:
            p.flush_buffer()

        self.assertGreater(particle1.velocity[0].value, 0)  # type: ignore
        self.assertLess(particle2.velocity[0].value, 0)  # type: ignore

        npt.assert_almost_equal(abs(particle1.velocity[0].value), abs(particle2.velocity[0].value))  # type: ignore

    def test_euler_integration_buffering(self):
        """Test that Euler integration properly uses the buffering system."""
        gravity_engine = Gravity(G=1.0)
        engines = [gravity_engine]  # type: ignore

        particle = Particle(
            mass=1.0 << u.kg,
            position=np.array([0.0, 0.0, 0.0]) << u.m,
            velocity=np.array([1.0, 0.0, 0.0]) << (u.m / u.s),  # type: ignore
            charge=0.0 << u.C,
        )
        particles = [particle]

        timestep = 0.1 << u.s
        initial_pos = particle.position.copy()
        initial_vel = particle.velocity.copy()

        Euler.integrate(engines, particles, timestep)  # type: ignore

        self.assertTrue("position" in particle.buffer)
        self.assertTrue("velocity" in particle.buffer)

        npt.assert_array_equal(particle.position.value, initial_pos.value)
        npt.assert_array_equal(particle.velocity.value, initial_vel.value)

        expected_new_pos = initial_pos + initial_vel * timestep
        npt.assert_array_almost_equal(particle.buffer["position"].value, expected_new_pos.value)  # type: ignore

    def test_gravity_symmetric_vs_asymmetric(self):
        """Test that gravity behaves symmetrically."""
        gravity_sym = Gravity(G=1.0)

        class AsymmetricGravity(Engine):
            _symmetric: bool = False

            def __init__(self, G: float = 1.0):
                super().__init__()
                self.G = G

            def force(self, particle: Particle, effector: Particle) -> u.Quantity:
                r_vec = effector.position - particle.position
                r_mag = np.linalg.norm(r_vec)
                r_direction = r_vec / r_mag
                f_mag = self.G * particle.mass.value * effector.mass.value / (r_mag.value ** 2)  # type: ignore
                return f_mag * r_direction << u.N

        gravity_asym = AsymmetricGravity(G=1.0)

        particle1 = Particle(
            mass=1.0 << u.kg,
            position=np.array([-0.5, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),  # type: ignore
            charge=0.0 << u.C,
        )
        particle2 = Particle(
            mass=1.0 << u.kg,
            position=np.array([0.5, 0.0, 0.0]) << u.m,
            velocity=np.array([0.0, 0.0, 0.0]) << (u.m / u.s),  # type: ignore
            charge=0.0 << u.C,
        )
        particles = [particle1, particle2]

        forces_sym = gravity_sym.interact(particles)
        forces_asym = gravity_asym.interact(particles)

        npt.assert_array_almost_equal(forces_sym.value, forces_asym.value, decimal=10)


if __name__ == "__main__":
    main()
