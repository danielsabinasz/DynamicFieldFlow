from dff.simulation import Simulator
from dfpy import connect, Field, CustomInput, GaussWeightPattern, TimedCustomInput, Node
import matplotlib.pyplot as plt

timed_custom_input = TimedCustomInput(
    timed_custom_input=[7.0]*5 + [0.0]*5
)

node = Node(
    resting_level=-5.0
)

connect(timed_custom_input, node)


sim = Simulator(record_values=True)
vals = []
for t in range(20, 200, 20):
    sim.simulate_until(t)
    vals.append(sim.get_value(node).numpy())
plt.plot(vals)
plt.show()

"""timed_custom_input.pattern = [0.0]*31+[6.0]*10+[0.0]*10

sim.update_initial_values()
sim.reset_time()
sim.simulate_until(1000)
plot = default_snapshot_plot(field)
plot.draw(sim.get_value(field))
plot.figure.show()"""
