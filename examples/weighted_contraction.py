from dff.simulation import Simulator
from dff.visualization import default_snapshot_plot
from dfpy import connect, GaussInput, Field, Dimension, SumWeightPattern, GaussWeightPattern, Sigmoid

gauss = GaussInput(
    dimensions=[Dimension(-25, 25, 51), Dimension(-15, 15, 31)],
    mean=[5.0, 10.0],
    height=5.0,
    sigmas=[3.0, 3.0]
)

field = Field(
    dimensions=[Dimension(-25, 25, 51)],
    resting_level=-5.0,
    interaction_kernel=SumWeightPattern([
        GaussWeightPattern(height=0.0, sigmas=(3.0,)),
        GaussWeightPattern(height=-0.0, sigmas=(5.0,))
    ]),
    global_inhibition=-0.1
)

connect(gauss, field, pointwise_weights=6.0, activation_function=Sigmoid(1), contract_dimensions=[1],
        contraction_weights=[20*[0.0]+11*[0.1]]*51)

sim = Simulator()
sim.simulate_until(1000)
plot1 = default_snapshot_plot(gauss)
plot1.draw(sim.get_value(gauss))
plot1.figure.show()
plot2 = default_snapshot_plot(field)
plot2.draw(sim.get_value(field))
plot2.figure.show()
