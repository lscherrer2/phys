from datetime import datetime as dt, timedelta as td
from phys.obj import Object
from phys.eng import Engine

__all__ = ["Sim"]

class Simulation:
    def __init__ (self, engine: Engine, start: dt, end: dt, timestep: td, objs: list[Object] = []):
        self.timestep = timestep
        self.objects = objs
        self.engine = engine
        self.start = start
        self.end = end

    def step (self, timestep: td):
        self.engine.step(self.objects, timestep)

    def exec (self): 
        epoch = self.start
        while epoch <= self.end:
            timestep = min(self.end - epoch, self.timestep) 
            self.step(timestep)
            epoch += timestep


Sim = Simulation
