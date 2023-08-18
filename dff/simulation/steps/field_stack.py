import logging
from functools import reduce

import tensorflow as tf

from dff.simulation.convolution import convolve, convolve_classic_stacked, convolve_fft_stacked_old, \
    convolve_fft_stacked_new
from dff.simulation.util import compute_positional_grid
from dff.simulation.weight_patterns import weight_pattern_config_from_dfpy_weight_pattern, compute_weight_pattern_tensor


def field_stack_prepare_constants(step):
    domain = [[dimension.lower, dimension.upper] for dimension in step.fields[0].dimensions]
    shape = tuple([dimension.size for dimension in step.fields[0].dimensions])

    # Compute bin size
    bin_size_by_dimension = [ (dimension.upper - dimension.lower)/dimension.size for dimension in step.fields[0].dimensions ]
    bin_size = reduce((lambda x, y: x * y), bin_size_by_dimension)

    return {"domain": domain, "shape": shape, "bin_size": bin_size}


def field_stack_prepare_variables(step):
    if step.assignable:
        resting_levels = tf.Variable([field.resting_level for field in step.fields], name=step.name + ".resting_level", trainable=step.trainable, constraint=lambda x: tf.math.minimum(x, 0))
        time_scales = tf.Variable([field.time_scale for field in step.fields], trainable=step.trainable)
        sigmoid_betas = tf.Variable([field.activation_function.beta for field in step.fields], trainable=step.trainable)
        global_inhibitions = tf.Variable([field.global_inhibition for field in step.fields], name=step.name + ".global_inhibition", trainable=step.trainable, constraint=lambda x: tf.math.minimum(x, 0))
        noise_strengths = tf.Variable([field.noise_strength for field in step.fields], trainable=step.trainable)
    else:
        resting_levels = tf.constant([field.resting_level for field in step.fields], name=step.name + ".resting_level")
        time_scales = tf.constant([field.time_scale for field in step.fields])
        sigmoid_betas = tf.constant([field.activation_function.beta for field in step.fields])
        global_inhibitions = tf.constant([field.global_inhibition for field in step.fields], name=step.name + ".global_inhibition")
        noise_strengths = tf.constant([field.noise_strength for field in step.fields])

    domain = step.fields[0].domain()
    shape = step.fields[0].shape()

    kernel_domain = []
    for i in range(len(domain)):
        lower = domain[i][0]
        upper = domain[i][1]
        rng = upper - lower
        kernel_domain.append([
            -rng/2, rng/2
        ])

    interaction_kernel_weight_pattern_configs = []
    for field in step.fields:
        interaction_kernel_weight_pattern_config =\
            weight_pattern_config_from_dfpy_weight_pattern(field.interaction_kernel, kernel_domain, shape)
        interaction_kernel_weight_pattern_configs.append(interaction_kernel_weight_pattern_config)

    return {"resting_levels": resting_levels, "time_scales": time_scales,
            "sigmoid_betas": sigmoid_betas,
            "global_inhibitions": global_inhibitions, "noise_strengths": noise_strengths,
            "interaction_kernel_weight_pattern_configs": interaction_kernel_weight_pattern_configs}


def field_stack_prepare_time_and_variable_invariant_tensors(step, shape, domain):
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
    if step.fields[0].interaction_kernel is not None:
        rng = step.fields[0].interaction_kernel.ranges()
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
#@tf.function
def field_stack_compute_time_invariant_variable_variant_tensors(shape, interaction_kernel_positional_grid, resting_levels,
                                                          interaction_kernel_weight_pattern_configs):
    #resting_level_tensor = tf.ones(tuple([int(x) for x in shape])) * resting_level
    ones = tf.ones(tuple([int(x) for x in shape]))
    if len(shape) == 1:
        resting_levels_tensor = ones * tf.reshape(resting_levels,(-1,1))
    if len(shape) == 2:
        resting_levels_tensor = ones * tf.reshape(resting_levels,(-1,1,1))
    if len(shape) == 3:
        resting_levels_tensor = ones * tf.reshape(resting_levels,(-1,1,1,1))

    interaction_kernel_weight_pattern_tensors = []
    for interaction_kernel_weight_pattern_config in interaction_kernel_weight_pattern_configs:
        interaction_kernel_weight_pattern_tensor = compute_weight_pattern_tensor(interaction_kernel_weight_pattern_config, interaction_kernel_positional_grid)
        interaction_kernel_weight_pattern_tensors.append(interaction_kernel_weight_pattern_tensor)
    interaction_kernel_weight_pattern_tensors = [tf.expand_dims(x,axis=0) for x in interaction_kernel_weight_pattern_tensors]
    interaction_kernel_weight_patterns_tensor = tf.concat(interaction_kernel_weight_pattern_tensors, axis=0)
    return {"resting_levels_tensor": resting_levels_tensor, "interaction_kernel_weight_patterns_tensor": interaction_kernel_weight_patterns_tensor}

import matplotlib.pyplot as plt

import time

@tf.function(jit_compile=True)
#@tf.function
def field_stack_time_step(time_step_duration, shape, bin_size, time_scales, sigmoid_betas,
                    global_inhibitions, noise_strengths, resting_levels_tensor, interaction_kernel_weight_patterns_tensor,
                    inputs=None, activations=None):
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
    logging.debug(f"trace field_stack_time_step: time_step_duration={time_step_duration}, shape={shape}, bin_size={bin_size}")

    num_fields = activations.shape[0]

    minus_u = tf.multiply(-1.0, activations)

    if len(shape) == 1:
        sigmoid_betas_expanded = tf.expand_dims(sigmoid_betas,-1)
    if len(shape) == 2:
        sigmoid_betas_expanded = tf.expand_dims(tf.expand_dims(sigmoid_betas,-1),-1)
    if len(shape) == 3:
        sigmoid_betas_expanded = tf.expand_dims(tf.expand_dims(tf.expand_dims(sigmoid_betas,-1),-1),-1)

    output = tf.math.sigmoid(tf.multiply(sigmoid_betas_expanded, activations))

    if len(output[0].shape) == 1:
        output_by_field = tf.reduce_sum(output, axis=1)
    if len(output[0].shape) == 2:
        output_by_field = tf.reduce_sum(output, axis=(1,2))
    if len(output[0].shape) == 3:
        output_by_field = tf.reduce_sum(output, axis=(1,2,3))

    total_inhibition_by_field = global_inhibitions * output_by_field
    if len(shape) == 1:
        total_inhibition_by_field_reshaped = tf.expand_dims(total_inhibition_by_field,-1)
    if len(shape) == 2:
        total_inhibition_by_field_reshaped = tf.expand_dims(tf.expand_dims(total_inhibition_by_field,-1),-1)
    if len(shape) == 3:
        total_inhibition_by_field_reshaped = tf.expand_dims(tf.expand_dims(tf.expand_dims(total_inhibition_by_field,-1),-1),-1)
    ones = tf.ones((num_fields,)+shape)
    global_inhibition_result = total_inhibition_by_field_reshaped * ones

    raw_noise = tf.random.normal((num_fields,) + shape)
    noise_term = tf.multiply(tf.multiply(tf.sqrt(time_step_duration), noise_strengths[0]), raw_noise) # TODO handle noise strengths appropriately

    #conv_result = convolve_fft_stacked_new(output, interaction_kernel_weight_patterns_tensor)
    conv_result = convolve_classic_stacked(output, interaction_kernel_weight_patterns_tensor[0])

    #tf.print("conv lat", output.shape, interaction_kernel.shape)
    sum = tf.add_n([minus_u, resting_levels_tensor, inputs, conv_result, global_inhibition_result, noise_term])

    expanded_time_scales = tf.reshape(time_scales, [-1] + [1] * (len(sum.shape) - 1))
    rate_of_change = tf.divide(sum, expanded_time_scales)

    result = tf.add(activations, tf.multiply(time_step_duration, rate_of_change))

    return result
