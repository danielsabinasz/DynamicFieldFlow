import logging
from functools import reduce

import tensorflow as tf

from dff.simulation.convolution import convolve
from dff.simulation.util import compute_positional_grid
from dff.simulation.weight_patterns import weight_pattern_config_from_dfpy_weight_pattern, compute_weight_pattern_tensor


def field_prepare_constants(step):
    domain = [[dimension.lower, dimension.upper] for dimension in step.dimensions]
    shape = tuple([dimension.size for dimension in step.dimensions])

    # Compute bin size
    bin_size_by_dimension = [ (dimension.upper - dimension.lower)/dimension.size for dimension in step.dimensions ]
    bin_size = reduce((lambda x, y: x * y), bin_size_by_dimension)

    return {"domain": domain, "shape": shape, "bin_size": bin_size}


def field_prepare_variables(step):
    if step.assignable:
        resting_level = tf.Variable(step.resting_level, name=step.name + ".resting_level", trainable=step.trainable, constraint=lambda x: tf.math.minimum(x, 0))
        time_scale = tf.Variable(step.time_scale, trainable=step.trainable)
        sigmoid_beta = tf.Variable(step.activation_function.beta, trainable=step.trainable)
        global_inhibition = tf.Variable(step.global_inhibition, name=step.name + ".global_inhibition", trainable=step.trainable, constraint=lambda x: tf.math.minimum(x, 0))
        noise_strength = tf.Variable(step.noise_strength, trainable=step.trainable)
    else:
        resting_level = tf.constant(step.resting_level, name=step.name + ".resting_level")
        time_scale = tf.constant(step.time_scale)
        sigmoid_beta = tf.constant(step.activation_function.beta)
        global_inhibition = tf.constant(step.global_inhibition, name=step.name + ".global_inhibition")
        noise_strength = tf.constant(step.noise_strength)

    domain = step.domain()
    shape = step.shape()

    kernel_domain = []
    for i in range(len(domain)):
        lower = domain[i][0]
        upper = domain[i][1]
        rng = upper - lower
        kernel_domain.append([
            -rng/2, rng/2
        ])

    interaction_kernel_weight_pattern_config =\
        weight_pattern_config_from_dfpy_weight_pattern(step.interaction_kernel, kernel_domain, shape)

    return {"resting_level": resting_level, "time_scale": time_scale,
            "sigmoid_beta": sigmoid_beta,
            "global_inhibition": global_inhibition, "noise_strength": noise_strength,
            "interaction_kernel_weight_pattern_config": interaction_kernel_weight_pattern_config}


def field_prepare_time_and_variable_invariant_tensors(step, shape, domain):
    positional_grid = compute_positional_grid(shape, domain)

    kernel_domain = []
    for i in range(len(domain)):
        lower = domain[i][0]
        upper = domain[i][1]
        rng = upper - lower
        kernel_domain.append([
            -rng/2, rng/2
        ])

    interaction_kernel_positional_grid = compute_positional_grid(shape, kernel_domain)

    # Clipping
    if step.interaction_kernel is not None:
        rng = step.interaction_kernel.ranges()
        if rng is not None:
            if len(rng) == 1:
                w = shape[0]
                interaction_kernel_positional_grid = interaction_kernel_positional_grid[
                                                     w//2-rng[0][1] : w//2+rng[0][0]+1
                                                     ]
            if len(rng) == 2:
                w = shape[0]
                h = shape[1]
                interaction_kernel_positional_grid = interaction_kernel_positional_grid[
                                                     w//2-rng[0][1] : w//2+rng[0][0]+1,
                                                     h//2-rng[1][1] : h//2+rng[1][0]+1
                                                     ]
            if len(rng) == 3:
                w = shape[0]
                h = shape[1]
                d = shape[2]
                interaction_kernel_positional_grid = interaction_kernel_positional_grid[
                                                     w//2-rng[0][1] : w//2+rng[0][0]+1,
                                                     h//2-rng[1][1] : h//2+rng[1][0]+1,
                                                     d//2-rng[2][1] : d//2+rng[2][0]+1
                                                     ]

    return {"positional_grid": positional_grid, "interaction_kernel_positional_grid": interaction_kernel_positional_grid}

# TODO performance
@tf.function
def field_compute_time_invariant_variable_variant_tensors(shape, interaction_kernel_positional_grid, resting_level,
                                                          interaction_kernel_weight_pattern_config):
    resting_level_tensor = tf.ones(tuple([int(x) for x in shape])) * resting_level
    interaction_kernel_weight_pattern_tensor = compute_weight_pattern_tensor(interaction_kernel_weight_pattern_config,
                                                                              interaction_kernel_positional_grid)
    return {"resting_level_tensor": resting_level_tensor, "interaction_kernel_weight_pattern_tensor": interaction_kernel_weight_pattern_tensor}

import matplotlib.pyplot as plt

import time

#@tf.function
def field_time_step(time_step_duration, shape, bin_size, time_scale, sigmoid_beta,
                    global_inhibition, noise_strength, resting_level_tensor, interaction_kernel,
                    input=None, activation=None):
    """Computes a time step of the Field step.

    :param Tensor time_step_duration: duration of a time step in milliseconds
    :param Tensor shape: shape of the field
    :param Tensor time_scale: time scale of the field (parameter tau in the field dynamics)
    :param Tensor sigmoid_beta: beta parameter of the sigmoid
    :param Tensor bin_size: size (in feature space) of a bin
    :param Tensor resting_level_tensor: tensor containing the field activation at resting level
    :param Tensor interaction_kernel: lateral interaction kernel
    :param Tensor global_inhibition: global inhibition inside the field
    :param Tensor noise_strength: amplitude of Gauss white noise
    :param Tensor input: input to the field
    :param Tensor activation: field activation from the previous time step
    :return: Tensor: field activation
    """
    logging.debug(f"trace field_time_step: time_step_duration={time_step_duration}, shape={shape}, bin_size={bin_size}, time_scale={time_scale}, sigmoid_beta={sigmoid_beta}, global_inhibition={global_inhibition}, noise_strength={noise_strength}")

    minus_u = tf.multiply(-1.0, activation)

    output = tf.math.sigmoid(tf.multiply(sigmoid_beta, activation))

    global_inhibition * tf.reduce_sum(output)

    global_inhibition_result = tf.ones(shape) * global_inhibition * tf.reduce_sum(output)

    noise_term = tf.multiply(tf.multiply(tf.sqrt(time_step_duration), noise_strength), tf.random.normal(shape))

    conv_result = convolve(output, interaction_kernel) * bin_size

    #tf.print("conv lat", output.shape, interaction_kernel.shape)
    sum = tf.add_n([minus_u, resting_level_tensor, input, conv_result, global_inhibition_result, noise_term])

    rate_of_change = tf.divide(sum, time_scale)

    result = tf.add(activation, tf.multiply(time_step_duration, rate_of_change))

    return result
