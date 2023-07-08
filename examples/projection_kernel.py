from dff.simulation import Simulator
from dff.visualization import default_snapshot_plot
from dfpy import connect, GaussInput, Field, GaussWeightPattern, Sigmoid
from dfpy.dimension import Dimension

gauss = GaussInput(
    dimensions=[Dimension(0, 50, 51)],
    mean=40.0,
    height=10.0,
    sigmas=(3.0,)
)

field = Field(
    dimensions=[Dimension(0, 50, 51)],
    resting_level=-5.0,
    interaction_kernel=GaussWeightPattern(height=1.0, sigmas=(3.0,)),
    global_inhibition=-1.2,
    noise_strength=0.2
)

connect(gauss, field, GaussWeightPattern(height=3.0, sigmas=3.0, mean=0.0), activation_function=Sigmoid(beta=1))

sim = Simulator()
sim.simulate_until(1000)
plot = default_snapshot_plot(field)
plot.draw(sim.get_value(field))
plot.figure.show()
