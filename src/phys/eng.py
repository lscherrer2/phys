from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from phys.obj import Object
from datetime import timedelta as td

__all__ = ["Engine", "Celestial"]

class Engine(ABC):

    def __init__ (self, objs: list[Object] = []):
        self.objs = objs

    def add_object (self, obj: Object):
        self.objs.append(obj)
    
    def add_objects (self, objs: list[Object]):
        self.objs.extend(objs)

    @abstractmethod
    def step (self, effectors: list[Object], timestep: td):
        pass


class Celestial (Engine):
    def __init__ (self, objs: list[Object], G: float = 1.0):
        super().__init__(objs)
        self.G = G

    def step (self, effectors: list[Object], timestep: td):
        for obj in self.objs:

            # amount obj.vel changes this step
            v_step: NDArray = np.array([0.0, 0.0, 0.0])

            for eff in effectors:

                # objects do not affect themselves
                if obj is eff:
                    continue
                
                # calculate change in velocity
                delta_v = (
                    timestep.seconds
                    * self.G 
                    * eff.mass 
                    / Object.distance(obj, eff)
                )

                # update velocity in correct direction
                v_step += delta_v * np.linalg.norm(eff.pos - obj.pos)

            # step position and velocity
            obj.pos += (obj.vel + 0.5 * v_step) * timestep.seconds
            obj.vel += v_step
            
                



