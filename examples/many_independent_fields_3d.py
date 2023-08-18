from dff.simulation import Simulator
from dff.visualization import default_snapshot_plot
from dfpy import connect, GaussInput, Field, SumWeightPattern, GaussWeightPattern
from dfpy.dimension import Dimension
import time

"""gauss = GaussInput(
    dimensions=[Dimension(-25, 25, 100), Dimension(-25, 25, 100)],
    height=7.0,
    sigmas=[5.0,5.0,]
)"""

fields = []

for i in range(100):
    field = Field(
        dimensions=[Dimension(-25, 25, 50), Dimension(-25, 25, 50), Dimension(-25, 25, 50)],
        resting_level=-5.0,
        interaction_kernel=GaussWeightPattern(height=1.0, sigmas=(2.0, 2.0, 2.0), field_size=(50,50,50)),
        global_inhibition=-0.1
    )
    fields.append(field)

sim = Simulator()

in_multiples_of = 1

sim.simulate_time_steps(in_multiples_of, in_multiples_of=in_multiples_of)
before = time.time()
sim.simulate_time_steps(in_multiples_of, in_multiples_of=in_multiples_of)
print("duration", str(1000*(time.time()-before)//in_multiples_of))
