from dff.simulation import Simulator
from dff.visualization import default_snapshot_plot
from dfpy import connect, GaussInput, Field, GaussWeightPattern
from dfpy.dimension import Dimension

gauss_1 = GaussInput(
    dimensions=[Dimension(-25, 25, 51), Dimension(-25, 25, 51)],
    mean=[-15.0, -15],
    height=9.0,
    sigmas=(5.0,5.0,)
)

gauss_2 = GaussInput(
    dimensions=[Dimension(-25, 25, 51), Dimension(-25, 25, 51)],
    mean=[15.0, 15.0],
    height=9.0,
    sigmas=(5.0,5.0,)
)

field = Field(
    dimensions=[Dimension(-25, 25, 51), Dimension(-25, 25, 51)],
    resting_level=-5.0,
    interaction_kernel=GaussWeightPattern(height=2.0, sigmas=(3.0, 3.0,)),
    global_inhibition=-1.5,
    noise_strength=0.2
)

connect(gauss_1, field)
connect(gauss_2, field)

sim = Simulator()
sim.simulate_until(1200)
plot = default_snapshot_plot(field)
plot.draw(sim.get_value(field))
plot.figure.show()
