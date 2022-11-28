from dfpy import connect, Node, TimedBoost
from dff import Simulator

num_ordinal_nodes = 5

ordinal_nodes = []
memory_nodes = []

for i in range(num_ordinal_nodes):
    # Create ordinal node
    ordinal_node = Node(
        resting_level=-5.0,
        self_excitation=8.0,
        name="Ordinal node " + str(i)
    )
    ordinal_nodes.append(ordinal_node)

    # Create memory node
    memory_node = Node(
        resting_level=-5.0,
        self_excitation=8.0,
        name="Memory node " + str(i)
    )
    memory_nodes.append(memory_node)

    connect(ordinal_node, memory_node, pointwise_weights=8.0)

    # Connect previous memory node to current ordinal node
    if i > 0:
        connect(memory_nodes[i - 1], ordinal_node, pointwise_weights=8.0)

    # Slightly inhibit ordinal node by memory node so that
    # it does not win the next competition
    connect(memory_node, ordinal_node, pointwise_weights=-2.0)

# Let ordinal nodes mutually inhibit each other
for i in range(num_ordinal_nodes):
    for j in range(num_ordinal_nodes):
        if i == j:
            continue
        connect(ordinal_nodes[i], ordinal_nodes[j], pointwise_weights=-8.0)


# Create static input to first ordinal node
static_input = TimedBoost(
    values={
        0: 8.0,
        200: 0.0
    }
)
connect(static_input, ordinal_nodes[0])

# Create proceed node
proceed_node = TimedBoost(
    values={
        0: 0.0,
        600: -12.0,

        1200: 0.0,
        1800: -12.0,

        2400: 0.0,
        3000: -12.0,

        3600: 0.0,
        4200: -12.0,

        4800: 0.0,
        5400: -12.0
    }
)
for i in range(num_ordinal_nodes):
    connect(proceed_node, ordinal_nodes[i])


sim = Simulator()
num_time_steps = int(6000/sim.time_step_duration)
for t in range(1, num_time_steps):
    sim.simulate_time_step()
    if t % 10 == 0:
        print([sim.get_value(o).numpy() for o in ordinal_nodes])
