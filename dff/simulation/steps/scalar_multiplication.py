import logging

import tensorflow as tf


def scalar_multiplication_prepare_constants(step):
    shape = tf.convert_to_tensor(step.shape, dtype=tf.int32)

    return {"shape": shape}


def scalar_multiplication_prepare_variables(step):
    scalar = tf.Variable(step.scalar, constraint=lambda x : tf.math.maximum(x, 0.0))
    return {"scalar": scalar}


@tf.function
def scalar_multiplication_time_step(scalar, input=None, activation=None):
    """Computes a time step of the ScalarMultiplication step.

    :param Tensor scalar: scalar to multiply the input with
    :param float input: input that should be multiplied with the scalar
    :return Tensor: result of the multiplication
    """
    logging.debug("trace scalar_multiplication_time_step")
    return tf.multiply(scalar, input)
