from dff.simulation import Simulator
from dfpy import connect, Field, CustomInput, GaussWeightPattern, TimedCustomInput, Node, Dimension, SumWeightPattern
import matplotlib.pyplot as plt

timed_custom_input = TimedCustomInput(
    timed_custom_input=[8.0]*15 + [8.0]*15
)

field = Field(
    dimensions=[Dimension(0, 50, 51)],
    resting_level=-5.0,
    interaction_kernel=SumWeightPattern([
        GaussWeightPattern(height=0.0, sigmas=(3.0,)),
        GaussWeightPattern(height=0.0, sigmas=(5.0,))
    ]),
    global_inhibition=0.0,
)

connect(timed_custom_input, field)


sim = Simulator(record_values=True)
vals = []
for t in range(20, 600, 20):
    print(t)
    sim.simulate_until(t)
    field_activation = sim.get_value(field).numpy()
    plt.plot(field_activation)
    plt.show()
    vals.append(field_activation)
#plt.plot(vals)
#plt.show()

"""timed_custom_input.pattern = [0.0]*31+[6.0]*10+[0.0]*10

sim.update_initial_values()
sim.reset_time()
sim.simulate_until(1000)
plot = default_snapshot_plot(field)
plot.draw(sim.get_value(field))
plot.figure.show()"""
