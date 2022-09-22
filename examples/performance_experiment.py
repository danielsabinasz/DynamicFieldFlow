from dfpy import NeuralStructure, GaussInput, Dimension, Field, GaussWeightPattern
import time
import logging
logging.basicConfig(level=logging.INFO)

from dff import Simulator

for x in range(50, 1010, 10):
    architecture = NeuralStructure()

    gauss_1 = GaussInput(
        dimensions=[Dimension(-25, 25, 51), Dimension(-15, 15, 31)],
        height=5.5,
        sigmas=[3.0, 3.0]
    )

    field_1 = Field(
        dimensions=[Dimension(-25, 25, 51), Dimension(-15, 15, 31)],
        resting_level=-5.0,
        interaction_kernel=GaussWeightPattern(height=1.0, sigmas=[3.0, 3.0]),
        global_inhibition=-0.01
    )

    architecture.connect(gauss_1, field_1)

    # Simulate
    simulator = Simulator(architecture, time_step_duration=10, default_simulation_call_type="largest")

    simulator.simulate_time_steps(100, in_multiples_of=100)

    before = time.time()
    print("GO")
    simulator.simulate_time_steps(100, in_multiples_of=100)
    print("DONE")
    duration = time.time() - before

    print(x, "\t", 1000*duration)
