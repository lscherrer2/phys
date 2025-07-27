from phys.integrators.integrator import Integrator
from phys.particle import Particle
from phys.forces.engine import Engine
from astropy.units import Quantity
from copy import deepcopy

__all__ = [
    "Leapfrog",
    "Yoshida4",
    "Yoshida6",
    "Yoshida8",
]


class Symplectic(Integrator):
    w_values: list[float]

    def integrate(
        self,
        engines: list[Engine],
        particles: list[Particle],
        timestep: Quantity,
    ):
        def symplectic_step(dummies: list[Particle], h: Quantity):
            # Compute initial accelerations
            accels = Integrator.accelerations(engines, dummies)

            # Perform initial half-step and position update
            for dummie, accel in zip(dummies, accels):
                dummie.velocity += 0.5 * h * accel
                dummie.position += h * dummie.velocity

            # Recompute accelerations
            accels = Integrator.accelerations(engines, dummies)

            # Perform second half-step
            for dummie, accel in zip(dummies, accels):
                dummie.velocity += 0.5 * h * accel

        dummy_particles = deepcopy(particles)

        # Step through each w
        symmetric_w_values = self.w_values + list(reversed(self.w_values[:-1]))
        for w in symmetric_w_values:
            h = timestep * w
            symplectic_step(dummy_particles, h)

        for particle, dummy in zip(particles, dummy_particles):
            particle.buffer["position"] = dummy.position
            particle.buffer["velocity"] = dummy.velocity


class Leapfrog(Symplectic):
    w_values = [
        1.0,
    ]


class Yoshida4(Symplectic):
    w_values = [
        1 / (2 - 2 ** (1 / 3)),
        1 - 2 * (1 / (2 - 2 ** (1 / 3))),
    ]


class Yoshida6(Symplectic):
    w_values = [
        -1.17767998417887,
        0.235573213359357,
        0.784513610477560,
    ]


class Yoshida8(Symplectic):
    w_values = [
        0.311790812418427,
        -1.55946803821447,
        -1.67896928259640,
        1.66335809963315,
        -1.06458714789183,
        1.36934946416871,
        0.629030650210433,
    ]
