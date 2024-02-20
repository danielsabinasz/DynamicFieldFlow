from dff.simulation import Simulator
from dfpy import Field, GaussWeightPattern, connect, GaussInput, Node
from dfpy.dimension import Dimension
from dff.visualization import default_snapshot_plot

gauss_input = GaussInput(
    dimensions=[Dimension(-25, 25, 51), Dimension(-25, 25, 51)],
    mean=[8.0, 5.0],
    height=8.0,
    sigmas=[3.0, 3.0]
)


fields = []

for i in range(100):
    field = Field(
        dimensions=[Dimension(-25, 25, 51), Dimension(-25, 25, 51)],
        resting_level=-5.0,
        interaction_kernel=GaussWeightPattern(height=1.0, sigmas=(2.0, 2.0,), field_size=(51,51)),
        global_inhibition=-0.1
    )
    fields.append(field)

    connect(gauss_input, field)


second_fields = []

for i in range(100):
    second_field = Field(
        dimensions=[Dimension(-25, 25, 51), Dimension(-25, 25, 51)],
        resting_level=-5.0,
        interaction_kernel=GaussWeightPattern(height=1.0, sigmas=(2.0, 2.0,), field_size=(51,51)),
        global_inhibition=-0.01
    )
    second_fields.append(second_field)
    connect(fields[i], second_fields[i], pointwise_weights=GaussWeightPattern(height=20.0, sigmas=(3.,3.)))


"""nodes = []
for i in range(100):
    node = Node(
        resting_level=-5.0
    )
    nodes.append(node)
    connect(second_fields[i], node, contract_dimensions=[0,1])"""





sim = Simulator(time_step_duration=20)

sim.simulate_time_steps(100)

plot = default_snapshot_plot(second_field)
plot.draw(sim.get_value(second_field).numpy())
plot.figure.show()

#print(sim.get_value(nodes[i]))