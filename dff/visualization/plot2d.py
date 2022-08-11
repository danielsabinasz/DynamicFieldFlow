import matplotlib.pyplot as plt
import tensorflow as tf

import dff.visualization
from dff.visualization.plot import Plot
from dfpy.steps import Step


class Plot2D(Plot):
    def __init__(self, step: Step):
        super().__init__(step)
        figure, axes = plt.subplots(figsize=(3,3))
        axes.set_title(step.name)
        self._figure = figure
        self._axes = axes
        self._pos = None
        self._value_range = [-10, 10]

    @property
    def value_range(self):
        return self._value_range

    @value_range.setter
    def value_range(self, value_range):
        self._value_range = value_range

    def draw(self, value):
        dimensions = self._step.dimensions

        if self._pos is None:
            vmin = self._value_range[0]
            vmax = self._value_range[1]
            self._pos = self._axes.imshow(tf.transpose(value),
                            cmap=dff.visualization.color_maps["parula"],
                            origin='lower',
                            extent=[dimensions[0].lower, dimensions[0].upper, dimensions[1].lower, dimensions[1].upper],
                            vmin=vmin,
                            vmax=vmax,
                            )
            self._colorbar = self._figure.colorbar(self._pos)
        else:
            self._pos.set_data(tf.transpose(value))
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()
