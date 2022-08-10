from dff.simulation import Simulator
from dff.visualization import default_snapshot_plot
from dfpy import connect, Field, CustomInput, GaussWeightPattern
from dfpy.dimension import Dimension

custom_input = CustomInput(pattern=[0.0]*10+[6.0]*10+[0.0]*31)

field = Field(
    dimensions=[Dimension(-25, 25, 51)],
    resting_level=-5.0,
    interaction_kernel=GaussWeightPattern(height=1.0, sigmas=(3.0,)),
    global_inhibition=-1.2,
    noise_strength=0.2
)

connect(custom_input, field)


sim = Simulator()
sim.simulate_until(1000)
plot = default_snapshot_plot(field)
plot.draw(sim.get_value(field))
plot.figure.show()
