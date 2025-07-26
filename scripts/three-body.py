from phys import Gravity, Simulation, Particle, Leapfrog
from astropy.units import kg, m, s, C

# Three particles
particles = [
    Particle(
        mass=1.0 << kg,
        charge=0.0 << C,
        position=[-1, 0, 0] << m,
        velocity=[0, -0.5, 0] << m / s,
    ),
    Particle(
        mass=1.0 << kg,
        charge=0.0 << C,
        position=[1, 0, 0] << m,
        velocity=[0, 0.5, 0] << m / s,
    ),
    Particle(
        mass=1.0 << kg,
        charge=0.0 << C,
        position=[1, 1, 0] << m,
        velocity=[0.28, 0.5, -0.38] << m / s,
    ),
]

# Gravity-like force
gravity = Gravity(1.0)

# Leapfrog integrator for beter KE preservation
leapfrog = Leapfrog()

# Set up and run the simulation
sim = Simulation(
    engines=[gravity],
    particles=particles,
    integrator=leapfrog,
)
sim.simulate(
    sim_time=20.0 << s,
    timestep=0.01 << s,
    record=True,
    verbose=True,
)

# Plot particle paths
sim.plot()
