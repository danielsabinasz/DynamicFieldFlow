from dff.simulation import Simulator
from dff.visualization import default_snapshot_plot
from dfpy import connect, Field, Node, GaussWeightPattern
from dfpy.dimension import Dimension

node = Node(resting_level=1.0)

field = Field(
    dimensions=[Dimension(-25, 25, 51)],
    resting_level=-5.0,
    interaction_kernel=GaussWeightPattern(height=1.0, sigmas=(3.0,)),
    global_inhibition=0.0,
    noise_strength=0.2
)

# Note: The pointwise_weights parameter can also be a numpy array or a TensorFlow tensor
connect(node, field, pointwise_weights=[6.0]*31 + [0.0]*20)


sim = Simulator()
sim.simulate_until(1000)
plot = default_snapshot_plot(field)
plot.draw(sim.get_value(field))
plot.figure.show()
