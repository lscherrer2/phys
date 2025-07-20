from phys import Engine, Particle
from typing import Optional
import numpy as np
import polars as pl

__all__ = ["Simulation"]

class Timer:
    def __init__ (self, timestep: float, end: float, start: float = 0):
        self.timestep = timestep
        self.end = end
        self.time = start
        self.done = self.time == end

    def __next__ (self):
        if self.time >= self.end:
            raise StopIteration()
        previous_time = self.time
        next_time = min(self.end, self.time + self.timestep)
        self.time = next_time
        return next_time - previous_time

    def __iter__ (self):
        return self

class Simulation:

    def __init__ (
        self,
        timestep: float,
        particles: Optional[list[Particle]],
        engines: Optional[list[Engine]],
    ):
        self.timestep: float = timestep
        self.particles: list[Particle] = particles if particles is not None else []
        self.engines: list[Engine] = engines if engines is not None else []
        self.db = []

    def add_particle (self, particle: Particle):
        self.particles.append(particle)

    def add_particles (self, particles: list[Particle]):
        for particle in particles:
            self.add_particle(particle)

    def step (self, timestep: Optional[float] = None):
        # use default timestep unless otherwise specified
        timestep = self.timestep if timestep is None else timestep

        # initialize future position
        for particle in self.particles:
            particle.buffer["position"] = particle.position + particle.velocity * timestep
            particle.buffer["velocity"] = particle.velocity.copy()

        # calculate forces on each particle
        for particle in self.particles:

            # calculate forces on particle
            forces: list[np.ndarray] = []
            for engine in self.engines:
                engine_forces = engine.batch_interact(particle, self.particles, cores=4).values()
                forces.extend(engine_forces)

            # calculate acceleration over timestep
            acceleration = sum(forces) / particle.mass

            # calculate deltas based on acceleration
            delta_v = acceleration * timestep
            delta_p = 0.5 * delta_v * timestep      # change in position due to acceleration

            # update particle buffers
            particle.buffer["position"] += delta_p
            particle.buffer["velocity"] += delta_v

        # flush buffers
        for particle in self.particles:
            particle.flush()

    def record (self, time: float):
        snapshot = {"time": time}
        snapshot |= {
            f"Particle {particle.num}": particle.position
            for particle in self.particles
        }
        self.db.append(snapshot)

    def reset_db (self):
        self.db = []

    @property
    def data (self):
        return pl.DataFrame(self.db)


    def simulate (self, end_time: float, timestep: Optional[float] = None):
        timestep = self.timestep if timestep is None else timestep
        timer = Timer(timestep, end_time, 0)
        self.record(timer.time)
        for timestep in timer:
            self.step(timestep)
            self.record(timer.time)



