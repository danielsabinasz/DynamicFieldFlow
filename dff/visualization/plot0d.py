import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

from dff import visualization
from dff.visualization.plot import Plot
from dfpy.steps import Step


class Plot0D(Plot):

    def __init__(self, step: Step):
        super().__init__(step)
        figure, axes = plt.subplots(figsize=(3,3))
        axes.set_title(step.name)
        axes.set_xticks([])
        axes.set_yticks([])
        self._figure = figure
        self._axes = axes
        self._line = None
        self._cmap = visualization.color_maps["parula"]
        self._marker_style = dict(linestyle=':', marker='o',
                            markersize=15, markerfacecoloralt='tab:red')
        self._vmin = -5
        self._vmax = 5

    def draw(self, value):
        if value < self._vmin:
            self._vmin = value
        if value > self._vmax:
            self._vmax = value
        value_normalized = (min(self._vmax, max(self._vmin, value)) - self._vmin) / (self._vmax - self._vmin)
        self._line, = self._axes.plot(0, 0, fillstyle='full', color=self._cmap(value_normalized), **(self._marker_style))
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()


def plot_0d(step: Step, value, width, height, activation_range=(-5.0, 5.0),
            cmap: Colormap=visualization.color_maps["parula"]):
    """Creates a snapshot plot of a step whose value corresponds to a scalar (e.g., the activation of a neural node).

    :param step: the step to visualize
    :param value: the value of the step
    :param width: width of the plot
    :param height: height of the plot
    :param activation_range: lower and upper bounds for color map
    :param cmap: color map
    """

    marker_style = dict(linestyle=':', marker='o',
                        markersize=15, markerfacecoloralt='tab:red')

    aspect_ratio = float(width)/height
    f = 3
    fig, ax = plt.subplots(figsize=(f, f/aspect_ratio), dpi=width/f)
    ax.set_title(step.name)

    if value is not None:
        value_normalized = (min(activation_range[1], max(activation_range[0], value)) - activation_range[0]) / (
                    activation_range[1] - activation_range[0])

        ax.plot(0, 0, fillstyle='full', color=cmap(value_normalized), **marker_style)

    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax
