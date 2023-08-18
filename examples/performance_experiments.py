from dff import visualization, Simulator
from dfpy import connect, Field, GaussInput, Dimension, GaussWeightPattern, initialize_architecture, SumWeightPattern
from dff.visualization import default_snapshot_plot
import time
import tensorflow as tf

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

for x in range(950, 960, 10):

    gauss = GaussInput(
        dimensions=[Dimension(-x/2, x/2, x), Dimension(-x/2, x/2, x)],
        height=5.5,
        sigmas=[3.0, 3.0]
    )

    field = Field(
        dimensions=[Dimension(-x/2, x/2, x), Dimension(-x/2, x/2, x)],
        resting_level=-5.0,
        interaction_kernel=GaussWeightPattern(height=1.0, sigmas=(2.0, 2.0,)),
        global_inhibition=-0.01
    )

    connect(gauss, field)

    # largest: 0.24 0.244 0.409 0.422 0.429 0.426
    # single: 1.37 1.41 1.44 1.56 1.46 1.67

    # Simulate
    simulator = Simulator(time_step_duration=10)
    simulator.simulate_time_steps(10, in_multiples_of=10)
    time_before = time.time()
    simulator.simulate_time_steps(2, in_multiples_of=10)
    simulator.get_value(field).numpy()
    duration = (time.time() - time_before)/10.0
    print("abholen", 1000 * duration)

    if x == 950:
        plot = default_snapshot_plot(field)
        plot.draw(simulator.get_value(field).numpy())
        plot.figure.show()

    initialize_architecture()

print("")