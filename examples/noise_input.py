from dff.simulation import Simulator
from dff.visualization import default_snapshot_plot
from dfpy import connect, Field, GaussWeightPattern, NoiseInput
from dfpy.dimension import Dimension
import matplotlib.pyplot as plt

noise = NoiseInput(
    dimensions=[Dimension(0, 300, 301)],
    strength=1.0
)

field = Field(
    dimensions=[Dimension(0, 300, 301)],
    time_scale=60,
    resting_level=-5.0,
    interaction_kernel=GaussWeightPattern(height=1.0, sigmas=(3.0,)),
    global_inhibition=-1.2,
    noise_strength=0.0
)

connect(noise, field, kernel_weights=GaussWeightPattern(height=0.1795, mean=0.0, sigmas=[2.0]))

sim = Simulator(time_step_duration=1)
for t in range(3):
    plot = default_snapshot_plot(noise)
    plot.draw(sim.get_value(noise))
    plt.title("Noise")
    plot.axes.set_ylim([-10, 10])
    plot.figure.show()
    plt.show()

    plot = default_snapshot_plot(field)
    plot.draw(sim.get_value(field))
    plt.title("Field")
    plot.axes.set_ylim([-10, 10])
    plot.figure.show()
    plt.show()

    sim.simulate_time_steps(1)


plot = default_snapshot_plot(noise)
plot.draw(sim.get_value(noise))
plt.title("Noise")
plot.axes.set_ylim([-10, 10])
plot.figure.show()
plt.show()

plot = default_snapshot_plot(field)
plot.draw(sim.get_value(field))
plt.title("Field")
plot.axes.set_ylim([-10, 10])
plot.figure.show()
plt.show()
