import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dff.visualization.plot import Plot
from dfpy import Step


class Plot1D(Plot):
    def __init__(self, step: Step):
        super().__init__(step)
        figure, axes = plt.subplots(figsize=(3,3))
        axes.set_title(step.name)
        self._figure = figure
        self._axes = axes
        self._line = None

        dimensions = self._step.dimensions
        self._linspace = np.linspace(dimensions[0].lower, dimensions[0].upper, dimensions[0].size, endpoint=True)

    def draw(self, value):
        if self._line is None:
            self._line, = self._axes.plot(self._linspace, value.numpy())
        else:
            #self._line, = self._axes.plot(self._linspace, value)
            self._line.set_ydata(value)
            self._axes.set_ylim(tf.reduce_min(value), tf.reduce_max(value))

        self._figure.canvas.draw()
        self._figure.canvas.flush_events()


def plot_1d(step, value, width, height):
    """Creates a snapshot plot of a step whose value corresponds to a 1d grid (e.g., the activation of a 1d field).

    :param steps: the step to visualize
    :param value: the value of the step
    :param width: width of the plot
    :param height: height of the plot
    """

    domain = step.domain
    shape = step.shape

    aspect_ratio = float(width)/height
    f = 3
    fig, ax = plt.subplots(figsize=(f, f/aspect_ratio), dpi=width/f)
    ax.set_title(step.name)

    if value is not None:
        linspace = np.linspace(domain[0][0], domain[0][1], shape[0], endpoint=True)
        ax.plot(linspace, value)

    return fig, ax
