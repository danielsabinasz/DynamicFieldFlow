from dff.simulation import Simulator
from dff.visualization import default_snapshot_plot
from dfpy import connect, GaussInput, Field, SumWeightPattern, GaussWeightPattern
from dfpy.dimension import Dimension

gauss = GaussInput(
    dimensions=[Dimension(-25, 25, 51)],
    mean=[0.0,],
    height=7.0,
    sigmas=[5.0,]
)

fields = []

for i in range(100):
    field = Field(
        dimensions=[Dimension(-25, 25, 51)],
        resting_level=-5.0,
        interaction_kernel=SumWeightPattern([
            GaussWeightPattern(height=1.0, sigmas=(3.0,)),
            GaussWeightPattern(height=-0.5, sigmas=(5.0,))
        ]),
        global_inhibition=-0.1
    )
    connect(gauss, field)
    fields.append(field)

sim = Simulator()

sim.simulate_until(1000)

plot = default_snapshot_plot(fields[0])
plot.draw(sim.get_value(fields[0]))
plot.figure.show()

# 5.75 1.60
# 5.70 1.58
# 5.69 1.60

# 5.83 1.64
# 5.82 1.60
# 5.86 1.63

# 6.95 2.19
#