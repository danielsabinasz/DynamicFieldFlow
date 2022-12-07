import tensorflow as tf
import numpy as np


def compute_positional_grid(shape: tf.Tensor, domain: list) -> tf.Tensor:
    """Computes a positional grid for the specified shape and domain, i.e., a tensor that specifies the location
    in feature space of each array entry of the specified shape.

    :param shape: shape of the field
    :param domain: domain over which the field is defined
    :return positional_grid: the positional grid
    """

    ndim = len(shape)

    # Assemble a set of linear spaces
    linspaces = []
    for i in range(0, ndim):
        linspaces.append(np.linspace(float(domain[i][0]),
                                     float(domain[i][1]),
                                     shape[i],
                                     endpoint=True,
                                     dtype=np.float32))

    # Combine linear spaces into meshgrid
    mgrid = tf.meshgrid(*linspaces, indexing="ij")

    # Reshape meshgrid
    positional_grid = np.zeros(tuple([int(x) for x in shape]) + (len(domain),), dtype=np.float32)

    if ndim == 1:
        for i in range(0, ndim):
            positional_grid[:, i] = mgrid[i][:]
    if ndim == 2:
        for i in range(0, ndim):
            positional_grid[:, :, i] = mgrid[i][:][:]
    if ndim == 3:
        for i in range(0, ndim):
            positional_grid[:, :, :, i] = mgrid[i][:][:][:]
    if ndim == 4:
        for i in range(0, ndim):
            positional_grid[:, :, :, :, i] = mgrid[i][:][:][:][:]

    return positional_grid

