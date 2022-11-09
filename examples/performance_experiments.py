from dff import visualization, Simulator
from dfpy import connect, Field, GaussInput, Dimension, GaussWeightPattern, initialize_architecture
import time

for x in range(50, 1010, 10):

    gauss = GaussInput(
        dimensions=[Dimension(-x/2, x/2, x+1-x%2), Dimension(-x/2, x/2, x+1-x%2)],
        height=5.5,
        sigmas=[3.0, 3.0]
    )

    field = Field(
        dimensions=[Dimension(-x/2, x/2, x+1-x%2), Dimension(-x/2, x/2, x+1-x%2)],
        resting_level=-5.0,
        interaction_kernel=GaussWeightPattern(height=1.0, sigmas=(3.0, 3.0)),
        global_inhibition=-0.01
    )

    connect(gauss, field)

    # largest: 0.24 0.244 0.409 0.422 0.429 0.426
    # single: 1.37 1.41 1.44 1.56 1.46 1.67

    # Simulate
    simulator = Simulator(time_step_duration=10, default_simulation_call_type="largest")
    simulator.simulate_time_steps(100)
    time_before = time.time()
    simulator.simulate_time_steps(100)
    duration = time.time() - time_before

    print(x, "\t", 1000 * duration/100)

    if x == 1000:
        fig = visualization.plot_2d(field, simulator.get_value(field), 300, 300)
        fig.show()

    initialize_architecture()
