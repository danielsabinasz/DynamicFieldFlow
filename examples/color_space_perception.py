from dff import Simulator, default_snapshot_plot
from dfpy import Field, Dimension, connect
from dff.input_processing.image import create_color_space_perception_field_input

dimensions = {
    "x": Dimension(-25, 25, name="x"),
    "y": Dimension(-25, 25, name="y"),
    "hue": Dimension(0, 25, name="hue", ticklabels={
        0: "red",
        4: "yellow",
        8: "green",
        16: "blue"
    })
}

color_space_perception_field = Field(
    name="color/space perception field",
    dimensions=[ dimensions["x"], dimensions["y"], dimensions["hue"] ],
    interaction_kernel="stabilized"
)

color_space_perception_field_input = create_color_space_perception_field_input(color_space_perception_field, "images/twoPair_fullPairB.jpg")

connect(color_space_perception_field_input, color_space_perception_field, pointwise_weights=7.0)

sim = Simulator()
sim.simulate_until(1000)
plot = default_snapshot_plot(color_space_perception_field)
plot.slice_values = [0, 4, 8, 16]
plot.draw(sim.get_value(color_space_perception_field))
plot.figure.show()
