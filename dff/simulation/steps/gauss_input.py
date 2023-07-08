import tensorflow as tf
import tensorflow_probability as tfp

from dff.simulation.util import compute_positional_grid
from dff.simulation.weight_patterns import compute_kernel_gauss_tensor, compute_kernel_gauss_tensor_with_positional_grid

tfd = tfp.distributions


def gauss_input_prepare_constants(step):
    domain = tf.convert_to_tensor([[dimension.lower, dimension.upper] for dimension in step.dimensions],
                                  dtype=tf.float32)
    shape = tf.convert_to_tensor([dimension.size for dimension in step.dimensions], dtype=tf.int32)
    return {"domain": domain, "shape": shape}


def gauss_input_prepare_variables(step):
    if step.assignable:
        height = tf.Variable(step.height, trainable=step.trainable)
        mean = tf.Variable(step.mean, trainable=step.trainable)
        sigmas = tf.Variable(step.sigmas, trainable=step.trainable)
    else:
        height = tf.constant(step.height)
        mean = tf.constant(step.mean)
        sigmas = tf.constant(step.sigmas)
    return {"height": height, "mean": mean, "sigmas": sigmas}


def gauss_input_prepare_time_and_variable_invariant_tensors(shape, domain):
    positional_grid = compute_positional_grid(shape, domain)

    return {"positional_grid": positional_grid}

"""def gauss_input_prepare_time_invariant_variable_variant_tensors(shape, domain, mean, sigmas, height, positional_grid):
    # Prepares constants for the Gauss step.

    :param Tensor shape: shape of the grid representation
    :param Tensor domain: domain over which the Gauss is defined
    :param Tensor mean: mean of the Gauss
    :param Tensor sigmas: sigmas of the Gauss
    :param Tensor height: height of the Gauss
    :param Tensor positional_grid
    :return Tensor gauss_input_tensor: a computed tensor representation of the Gauss

    gauss_input_tensor = compute_kernel_gauss_tensor_with_positional_grid(mean, sigmas, height, positional_grid)

    return gauss_input_tensor,"""

@tf.function
def gauss_input_time_step(height, mean, sigmas, positional_grid):
    return compute_kernel_gauss_tensor_with_positional_grid(mean, sigmas, height, positional_grid)
