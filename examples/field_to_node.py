from dff.simulation import Simulator
from dff.visualization import default_snapshot_plot
from dfpy import connect, GaussInput, Field, Dimension, SumWeightPattern, GaussWeightPattern, Sigmoid, Node

gauss = GaussInput(
    dimensions=[Dimension(-25, 25, 51)],
    mean=[10.0],
    height=5.0,
    sigmas=[3.0]
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

node = Node()

connect(gauss, field, pointwise_weights=6.0, activation_function=Sigmoid(1))
connect(field, node, contract_dimensions=[0], contraction_weights=[0.0]*30+[1.0]*21)

sim = Simulator()
sim.simulate_until(1000)

plot1 = default_snapshot_plot(field)
plot1.draw(sim.get_value(field))
plot1.figure.show()

print(sim.get_value(node))
