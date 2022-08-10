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

    def draw(self, value):
        dimensions = self._step.dimensions

        if self._pos is None:
            vmin = tf.reduce_min(value)
            vmax = tf.reduce_max(value)
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
