from dff import visualization, Simulator
from dfpy import connect, Field, GaussInput, Dimension, GaussWeightPattern, initialize_architecture, dimensions_from_sizes
import time
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gauss = GaussInput(
    dimensions=(49,49),
    height=5.5,
    mean=(25.0,25.0),
    sigmas=[3.0, 3.0]
)

fields_layer_1 = []

for n in range(20):

    field = Field(
        dimensions=(49,49),
        resting_level=-5.0,
        interaction_kernel=GaussWeightPattern(height=1.0, sigmas=(3.0, 3.0)),
        global_inhibition=-0.01
    )

    connect(gauss, field)
    fields_layer_1.append(field)


for n in range(20):

    field = Field(
        dimensions=(49,49),
        resting_level=-5.0,
        interaction_kernel=GaussWeightPattern(height=1.0, sigmas=(3.0, 3.0)),
        global_inhibition=-0.01
    )

    connect(fields_layer_1[n], field)



# Simulate
simulator = Simulator(time_step_duration=10, default_simulation_call_type="largest")
time_before = time.time()
simulator.simulate_time_steps(10, in_multiples_of=None, mode="single")
duration = time.time() - time_before
print("tracing", duration)
time_before = time.time()

n = 10

simulator.simulate_time_steps(n, in_multiples_of=None, mode="single")
duration = time.time() - time_before
print(duration/float(n))

