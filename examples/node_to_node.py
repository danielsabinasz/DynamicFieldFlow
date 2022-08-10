from dff.simulation import Simulator
from dfpy import connect, Node

node1 = Node(resting_level=1.0)

node2 = Node(resting_level=-5.0)

connect(node1, node2, pointwise_weights=6.0)

sim = Simulator()
sim.simulate_until(1000)
print(sim.get_value(node2))
