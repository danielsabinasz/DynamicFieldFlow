import logging

import tensorflow as tf


@tf.function
def spatial_relation_pattern(domain, shape, angle_mean=0.0, angle_stddev=0.25, radius_mean=15.0, radius_stddev=100.0):
    """Prepares constants for the spatial relation pattern step.

    :param array_like domain: domain over which the Gauss is defined
    :param int or tuple of ints shape: shape of the grid representation
    :param float angle_mean: mean angle
    :param float angle_stddev: standard deviation of the angle
    :param float radius_mean: mean radius
    :param float radius_stddev: standard deviation of the radius
    :return Tensor pattern: the spatial relation pattern
    """
    logging.debug("trace spatial_relation_pattern")

    if type(shape) == int:
        domain = [domain]
        shape = (shape,)

    domain = tf.constant(domain, dtype=tf.float32)

    linspace_x = tf.linspace(domain[0][0], domain[0][1], shape[0])
    linspace_y = tf.linspace(domain[1][0], domain[1][1], shape[1])

    X, Y = tf.meshgrid(linspace_x, linspace_y)

    angle = tf.atan2(Y, X)
    radius = tf.math.log(
        tf.math.sqrt(
            tf.math.pow(X, 2.0) + tf.math.pow(Y, 2.0)
        )
    )

    pattern = tf.math.exp(
        -0.5 * tf.math.pow(angle - angle_mean, 2.0) / tf.math.pow(angle_stddev, 2.0)
        - 0.5 * tf.math.pow(radius - radius_mean, 2.0) / tf.math.pow(radius_stddev, 2.0)
    )

    return pattern
