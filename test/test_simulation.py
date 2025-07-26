from unittest import TestCase, main
import astropy.units as u
from phys.simulation import Simulation
from phys.particle import Particle
from phys.forces.gravity import Gravity
from phys.integrators.euler import Euler


class TestSimulation(TestCase):
    def test_setup(self):
        engines = [Gravity(1.0), Gravity(2.0)]
        particles = [Particle.random() for _ in range(10)]
        integrator = Euler()
        sim = Simulation(
            engines=engines,
            particles=particles,
            integrator=integrator,
        )
        self.assertIs(sim.engines, engines)
        self.assertIs(sim.particles, particles)
        self.assertIs(sim.integrator, integrator)

    def test_simulate(self):
        engines = [Gravity(1.0), Gravity(2.0)]
        particles = [Particle.random() for _ in range(10)]
        integrator = Euler()
        sim = Simulation(
            engines=engines,
            particles=particles,
            integrator=integrator,
        )
        sim.simulate(sim_time=10.0 << u.s, timestep=1.0 << u.s)
        self.assertEqual(len(sim.snapshots), 11)


def observe_results():
    engines = [Gravity(1.0)]
    particles = [Particle.random() for _ in range(3)]
    for particle in particles:
        particle.velocity /= 3.0
    integrator = Euler()
    sim = Simulation(
        engines=engines,
        particles=particles,
        integrator=integrator,
    )
    sim.simulate(sim_time=10.0 << u.s, timestep=0.01 << u.s)
    sim.plot()


if __name__ == "__main__":
    observe_results()
    main()
