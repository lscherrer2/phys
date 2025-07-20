from unittest import TestCase, main
from phys import Simulation, Particle, Gravity, Electromagnetism
import numpy as np

class TestSimulation (TestCase):
    def test_sim_setup (self):
        particles = [
            Particle(
                mass=np.random.randn(1).item(),
                position=np.random.randn(3),
                velocity=np.random.randn(3),
                charge=np.random.randn(1).item(),
            )
            for _ in range (3)
        ]
        sim = Simulation(
            timestep = 0.1,
            particles=particles,
            engines=[Gravity(0.5), Electromagnetism(1.0)]
        )

    def test_sim_run (self):
        particles = [
            Particle(
                mass=np.random.randn(1).item(),
                position=np.random.randn(3),
                velocity=np.random.randn(3),
                charge=np.random.randn(1).item(),
            )
            for _ in range (3)
        ]
        sim = Simulation(
            timestep = 0.1,
            particles=particles,
            engines=[Gravity(0.5), Electromagnetism(1.0)]
        )
        sim.simulate(1.0, 0.1)
        print(sim.data)

if __name__ == "__main__":
    print("running file")
    main()



