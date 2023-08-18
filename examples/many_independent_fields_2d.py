from dff.simulation import Simulator
from dff.visualization import default_snapshot_plot
from dfpy import connect, GaussInput, Field, SumWeightPattern, GaussWeightPattern
from dfpy.dimension import Dimension
import time
import matplotlib.pyplot as plt

gauss = GaussInput(
    dimensions=[Dimension(-25, 25, 100), Dimension(-25, 25, 100)],
    height=7.0,
    sigmas=[5.0,5.0,]
)

fields = []

for i in range(100):
    field = Field(
        dimensions=[Dimension(-25, 25, 100), Dimension(-25, 25, 100)],
        resting_level=-5.0,
        interaction_kernel=GaussWeightPattern(height=1.0, sigmas=(2.0, 2.0), field_size=(100,100)),
        global_inhibition=-0.1
    )
    connect(gauss, field)
    fields.append(field)

sim = Simulator()

num_time_steps = 1
in_multiples_of = 1

sim.simulate_time_steps(num_time_steps, in_multiples_of=in_multiples_of)
before = time.time()
sim.simulate_time_steps(num_time_steps, in_multiples_of=in_multiples_of)
print("duration", str(1000*(time.time()-before)/num_time_steps))

plt.imshow(sim.get_value(field).numpy()) and plt.colorbar() and plt.show()