from phys.integrators.integrator import Integrator
from phys.particle import Particle
from phys.forces.engine import Engine
from astropy.units import Quantity
from copy import deepcopy

__all__ = ["Leapfrog"]


class Leapfrog(Integrator):
    @staticmethod
    def integrate(engines: list[Engine], particles: list[Particle], timestep: Quantity):
        # Use dummy particles to avoid flushing buffers on true particles
        dummy_particles = deepcopy(particles)
        accels = Integrator.accelerations(engines, dummy_particles)

        # Half-step velocity update & full-step position update
        for particle, accel in zip(dummy_particles, accels):
            particle.buffer["velocity"] = particle.velocity + 0.5 * accel * timestep
            particle.buffer["position"] = (
                particle.position + particle.buffer["velocity"] * timestep
            )
            particle.flush_buffer()

        # Compute new accelerations
        accels = Integrator.accelerations(engines, dummy_particles)

        # Half-step velocity update
        for particle, accel in zip(dummy_particles, accels):
            particle.buffer["velocity"] = particle.velocity + 0.5 * accel * timestep
            particle.flush_buffer()

        # Copy computed buffers into true particles
        for particle, dummy_particle in zip(particles, dummy_particles):
            particle.buffer["position"] = dummy_particle.position
            particle.buffer["velocity"] = dummy_particle.velocity
