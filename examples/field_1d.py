from dff.simulation import Simulator
from dff.visualization import default_snapshot_plot
from dfpy import connect, GaussInput, Field, GaussWeightPattern, SumWeightPattern
from dfpy.dimension import Dimension
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt

gauss = GaussInput(
    dimensions=[Dimension(-150, 150, 301)],
    mean=0.0,
    height=9.0,
    sigmas=(3.0,)
)

field = Field(
    dimensions=[Dimension(-150, 150, 301)],
    resting_level=-5.0,
    interaction_kernel=SumWeightPattern([
        GaussWeightPattern(height=4.0, sigmas=(3.0,)),
        GaussWeightPattern(height=-2., sigmas=(6.0,))
    ], field_size=301, cutoff_factor=4.),
    global_inhibition=-1.2,
    noise_strength=0.2
)

connect(gauss, field)

sim = Simulator()
sim.simulate_until(1)
sim.reset_time()
before = time.time()
sim.simulate_until(1000)
duration = time.time() - before
print(duration)
plot = default_snapshot_plot(field)
plot.draw(sim.get_value(field))
plot.figure.show()
