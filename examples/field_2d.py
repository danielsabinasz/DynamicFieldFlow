from dff.simulation import Simulator
from dff.visualization import default_snapshot_plot
from dfpy import connect, GaussInput, Field, Dimension, SumWeightPattern, GaussWeightPattern, Sigmoid

gauss = GaussInput(
    dimensions=[Dimension(-25, 25, 51), Dimension(-15, 15, 31)],
    mean=[7.0, 4.0],
    height=5.0,
    sigmas=[5.0, 5.0]
)

field = Field(
    dimensions=[Dimension(-25, 25, 51), Dimension(-15, 15, 31)],
    resting_level=-5.0,
    interaction_kernel=SumWeightPattern([
        GaussWeightPattern(height=1.0, sigmas=(3.0, 3.0)),
        GaussWeightPattern(height=-0.5, sigmas=(5.0, 5.0))
    ]),
    global_inhibition=-0.1
)

connect(gauss, field, pointwise_weights=6.0, activation_function=Sigmoid(1))

sim = Simulator()
sim.simulate_until(700)
plot = default_snapshot_plot(field)
plot.draw(sim.get_value(field))
plot.figure.show()
