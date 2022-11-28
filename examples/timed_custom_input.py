from dff.simulation import Simulator
from dff.visualization import default_snapshot_plot
from dfpy import connect, Field, CustomInput, GaussWeightPattern, TimedCustomInput
from dfpy.dimension import Dimension

timed_custom_input = TimedCustomInput(
    dimensions=[Dimension(-25, 25, 51)],
    timed_custom_input=
[ [0.0]*10+[6.0]*31+[0.0]*10 ] * 30 +
[ [0.0]*51 ] * 20
)

field = Field(
    dimensions=[Dimension(-25, 25, 51)],
    resting_level=-5.0,
    interaction_kernel=GaussWeightPattern(height=1.0, sigmas=(3.0,)),
    global_inhibition=0.0
)

connect(timed_custom_input, field)


sim = Simulator()

sim.simulate_until(1000)
plot = default_snapshot_plot(field)
plot.draw(sim.get_value(field))
plot.figure.show()

"""timed_custom_input.pattern = [0.0]*31+[6.0]*10+[0.0]*10

sim.update_initial_values()
sim.reset_time()
sim.simulate_until(1000)
plot = default_snapshot_plot(field)
plot.draw(sim.get_value(field))
plot.figure.show()"""
