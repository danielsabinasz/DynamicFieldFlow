import matplotlib.pyplot as plt
import numpy as np
from dff.visualization._color_maps import color_maps
from dff.visualization.plot import Plot
from dfpy.steps import Step
import tensorflow as tf
import matplotlib.cm as cm

class Plot3D(Plot):
    def __init__(self, step: Step):
        super().__init__(step)
        figure, axes = plt.subplots(subplot_kw={"projection": "3d"})
        axes.set_title(step.name)
        self._figure = figure
        self._axes = axes
        self._pos = None
        self._value_range = [-15, 15]
        self._slice_values = []

    @property
    def value_range(self):
        return self._value_range

    @value_range.setter
    def value_range(self, value_range):
        self._value_range = value_range

    @property
    def slice_values(self):
        return self._slice_values

    @slice_values.setter
    def slice_values(self, slice_values):
        self._slice_values = slice_values

    def draw(self, value):
        dimensions = self._step.dimensions

        extent = [dimensions[0].lower, dimensions[0].upper, dimensions[1].lower, dimensions[1].upper],

        vmin = self._value_range[0]
        vmax = self._value_range[1]
        width = value.shape[0]
        height = value.shape[1]

        for v in self._slice_values:
            matrix2d = value[:,:,v]
            matrix2d = tf.transpose(matrix2d, [1, 0])

            X, Y = np.meshgrid(np.linspace(dimensions[0].lower,dimensions[0].upper,width,False), np.linspace(dimensions[1].lower,dimensions[1].upper,height,False))
            Z = v * np.ones(shape=(height,width))

            self._axes.set_box_aspect([1, height/width, 1])
            matrix2d = (matrix2d-vmin)/(vmax-vmin)
            fc = color_maps["parula"](matrix2d)
            p = self._axes.plot_surface(X, Y, Z, rstride=1, cstride=1, shade=False, alpha=0.5, facecolors=fc, vmin=vmin, vmax=vmax)
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()
