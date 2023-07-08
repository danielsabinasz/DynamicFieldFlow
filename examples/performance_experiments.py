from dff import visualization, Simulator
from dfpy import connect, Field, GaussInput, Dimension, GaussWeightPattern, initialize_architecture, SumWeightPattern
from dff.visualization import default_snapshot_plot
import time

for x in range(500, 510, 10):

    gauss = GaussInput(
        dimensions=[Dimension(-x/2, x/2, x+1-x%2), Dimension(-x/2, x/2, x+1-x%2)],
        height=5.5,
        sigmas=[3.0, 3.0]
    )

    field = Field(
        dimensions=[Dimension(-x/2, x/2, x+1-x%2), Dimension(-x/2, x/2, x+1-x%2)],
        resting_level=-5.0,
        interaction_kernel=SumWeightPattern([
            GaussWeightPattern(height=0.4, sigmas=(2.0, 2.0,)),
            GaussWeightPattern(height=-0.11, sigmas=(4.0, 4.0,))
        ]),
        global_inhibition=-0.01
    )

    connect(gauss, field)

    # largest: 0.24 0.244 0.409 0.422 0.429 0.426
    # single: 1.37 1.41 1.44 1.56 1.46 1.67

    # Simulate
    simulator = Simulator(time_step_duration=10, default_simulation_call_type="largest")
    simulator.simulate_time_steps(1, in_multiples_of=None, mode="single")
    time_before = time.time()
    simulator.simulate_time_steps(100, in_multiples_of=None, mode="single")
    duration = (time.time() - time_before)/100.0

    print(x, "\t", 1000 * duration)

    if x == 500:
        plot = default_snapshot_plot(field)
        plot.draw(simulator.get_value(field).numpy())
        plot.figure.show()

    initialize_architecture()
