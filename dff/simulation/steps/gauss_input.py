import tensorflow as tf
import tensorflow_probability as tfp

from dff.simulation.weight_patterns import compute_kernel_gauss_tensor

tfd = tfp.distributions


def gauss_input_prepare_constants(step):
    domain = tf.convert_to_tensor([[dimension.lower, dimension.upper] for dimension in step.dimensions],
                                  dtype=tf.float32)
    shape = tf.convert_to_tensor([dimension.size for dimension in step.dimensions], dtype=tf.int32)
    return {"domain": domain, "shape": shape}


def gauss_input_prepare_variables(step):
    height = tf.Variable(step.height)
    mean = tf.Variable(step.mean)
    sigmas = tf.Variable(step.sigmas)
    return {"height": height, "mean": mean, "sigmas": sigmas}


def gauss_input_prepare_time_and_variable_invariant_tensors(shape, domain, mean, sigmas, height):
    """Prepares constants for the Gauss step.

    :param Tensor shape: shape of the grid representation
    :param Tensor domain: domain over which the Gauss is defined
    :param Tensor mean: mean of the Gauss
    :param Tensor sigmas: sigmas of the Gauss
    :param Tensor height: height of the Gauss
    :return Tensor gauss_input_tensor: a computed tensor representation of the Gauss
    """

    gauss_input_tensor = compute_kernel_gauss_tensor(shape, domain, mean, sigmas, height)

    return gauss_input_tensor,
