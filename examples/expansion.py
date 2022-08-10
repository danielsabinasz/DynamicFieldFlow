from dff.simulation import Simulator
from dff.visualization import default_snapshot_plot
from dfpy import connect, GaussInput, Field, Dimension, SumWeightPattern, GaussWeightPattern, Sigmoid

gauss = GaussInput(
    dimensions=[Dimension(-25, 25, 51)],
    mean=10.0,
    height=1.0,
    sigmas=(3.0,)
)

field = Field(
    dimensions=[Dimension(-25, 25, 51), Dimension(-15, 15, 31)],
    resting_level=-5.0,
    interaction_kernel=SumWeightPattern([
        GaussWeightPattern(height=0.0, sigmas=(3.0, 3.0)),
        GaussWeightPattern(height=-0.0, sigmas=(5.0, 5.0))
    ]),
    global_inhibition=0.0
)

connect(gauss, field, pointwise_weights=7.0, activation_function=Sigmoid(1), expand_dimensions=[1])

sim = Simulator()
sim.simulate_until(1000)
plot1 = default_snapshot_plot(gauss)
plot1.draw(sim.get_value(gauss))
plot1.figure.show()
plot2 = default_snapshot_plot(field)
plot2.draw(sim.get_value(field))
plot2.figure.show()
