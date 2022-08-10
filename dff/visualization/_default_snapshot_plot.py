import matplotlib.pyplot as plt

import dff.visualization
from dff.visualization.plot0d import Plot0D
from dff.visualization.plot1d import Plot1D
from dff.visualization.plot2d import Plot2D
from dfpy.steps import Field, Image, GaussInput, TimedBoost, Node


def default_snapshot_plot(step):
    """Creates a snapshot plot, i.e., a single image that displays the state of a step at a given moment in time.
    For each type of step, a default way of visualizing that step is chosen.

    :param step: the step to visualize
    """

    if type(step) == Field or type(step) == GaussInput:

        ndim = len(step.dimensions)

        if ndim == 1:
            return Plot1D(step)
        elif ndim == 2:
            return Plot2D(step)

    elif type(step) == Image:

        return dff.visualization.image_plot

    elif type(step) == TimedBoost or type(step) == Node:

        return Plot0D(step)

    else:

        #raise ValueError("No default snapshot plot specified for " + str(type(step)))

        return None
