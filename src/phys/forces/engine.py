from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable
import json
from threading import Thread

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from phys.entities.particle import Particle

__all__ = ["Engine"]

class Engine (ABC):

    def __init__ (self, **kwargs):
        print(f"Received unrecognized keyword arguments:\n", json.dumps(kwargs, indent=4))

    @abstractmethod
    def interact (self, particle: Particle, effector: Particle) -> NDArray:
        """Calculates the force between particles"""
        ...

    def batch_interact (
        self, 
        particle: Particle, 
        effectors: list[Particle], 
        cores: int = 1
    ) -> dict[Particle, NDArray]:

        num_interactions = effectors.__len__()
        if cores == 1:
            return { 
                effector: self.interact(particle, effector) 
                for effector in effectors 
            }

        def single_interaction (
            particle: Particle, 
            effector: Particle, 
            interact_fn: Callable,
            result_map: dict[Particle, NDArray]
        ):
            result_map |= { particle: interact_fn(particle, effector) }

        def multiple_interaction (
            particle: Particle, 
            effectors: list[Particle], 
            interact_fn: Callable,
            result_map: dict[Particle, NDArray]
        ):
            for effector in effectors:
                result_map |= { particle: interact_fn(particle, effector) }


        result = {}
        threads: list[Thread] = []

        if num_interactions > cores:
            increment = num_interactions // cores
            for core in range(cores):
                effector_batch = effectors[core*increment:min(core*(increment+1),num_interactions)]
                thread = Thread(
                    target=multiple_interaction,
                    kwargs={
                        'particle': particle,
                        'effectors': effector_batch,
                        'interact_fn': self.interact,
                        'result_map': result,
                    }
                )
                threads.append(thread)
 
        else:
            for effector in effectors:
                thread = Thread(
                    target=single_interaction, 
                    kwargs={
                        'particle': particle,
                        'effector': effector,
                        'interact_fn': self.interact,
                        'result_map': result,
                    }
                )
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()
        
        return result




        


