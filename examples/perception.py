from dff import Simulator, default_snapshot_plot
from dfpy import Field, Dimension, SumWeightPattern, GaussWeightPattern, connect, Sigmoid
from dff.input_processing.image import create_hue_space_perception_field_input
import matplotlib.pyplot as plt
import numpy as np

field = Field(
    name="color/space perception field",
    dimensions=[Dimension(-25, 25, 51), Dimension(-25, 25, 51), Dimension(0, 25, 25)],
    interaction_kernel=SumWeightPattern([
        GaussWeightPattern(height=0.4, sigmas=(2.0, 2.0, 0.1)),
        GaussWeightPattern(height=-0.11, sigmas=(4.0, 4.0, 0.1))
    ])
)

hue_space_perception_field_input = create_hue_space_perception_field_input(field, "images/twoPair_fullPairB.jpg")

connect(hue_space_perception_field_input, field, pointwise_weights=7.0)


sim = Simulator()
sim.simulate_until(1000)
plot = default_snapshot_plot(field)
plot.slice_values = [0] #, 4, 8, 16
plot.draw(sim.get_value(field))
plot.figure.show()
