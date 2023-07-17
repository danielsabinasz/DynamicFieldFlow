from dff.simulation import Simulator
from dff.visualization import default_snapshot_plot
from dfpy import connect, GaussInput, Field, Dimension, SumWeightPattern, GaussWeightPattern, Sigmoid

gauss = GaussInput(
    dimensions=[Dimension(-25, 25, 51), Dimension(-15, 15, 31), Dimension(0, 25, 25)],
    mean=[8.0, 5.0, 13.0],
    height=5.0,
    sigmas=[5.0, 5.0, 2.0]
)

field = Field(
    dimensions=[Dimension(-25, 25, 51), Dimension(-15, 15, 31), Dimension(0, 25, 25)],
    resting_level=-5.0,
    interaction_kernel=SumWeightPattern([
        GaussWeightPattern(height=0.4, sigmas=(2.0, 2.0, 2.0)),
        GaussWeightPattern(height=-0.11, sigmas=(4.0, 4.0, 4.0))
    ], field_size=(51,31,25)),
    global_inhibition=0.0
)


connect(gauss, field, pointwise_weights=6.0)

sim = Simulator()
sim.simulate_until(1000)
plot = default_snapshot_plot(field)
plot.slice_values = [0, 13, 24]
plot.draw(sim.get_value(field))
plot.figure.show()
