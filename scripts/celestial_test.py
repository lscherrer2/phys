import phys
from datetime import datetime as dt, timedelta as td
from phys import Celestial, Object, Sim

objs = [
    Object(
        mass = 10,
        pos = [0, 0.2, 0],
        vel = [0, 0.1, -0.4]
    ),
    Object(
        mass = 0.2,
        pos = [4.0, 2.0, 1.0],
        vel = [0, 2.0, 1.0]
    ),
    Object(
        mass = 3,
        pos = [0, 0.2, 4.0],
        vel = [1.5, 2.0, -0.4]
    )
]
sim = Sim(
    engine=Celestial(objs=objs),
    start=dt.now(),
    end=dt.now()+td(minutes=1),
    timestep=td(seconds=1)
)
sim.exec()
