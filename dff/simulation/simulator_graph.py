import time
import logging

import dfpy.config
from dff.simulation.convolution import convolve
from dff.simulation import steps

logger = logging.getLogger(__name__)

import tensorflow as tf
from dfpy.steps import *
import dff.simulation.steps


def create_unrolled_simulation_call(num_time_steps,
                                    time_step_duration,
                                    steps,
                                    input_step_indices_by_step_index,
                                    activation_function_types_by_step_index,
                                    activation_function_betas_by_step_index,
                                    connection_kernel_weights_by_step_index,
                                    connection_pointwise_weights_by_step_index,
                                    connection_contract_dimensions_by_step_index,
                                    connection_contraction_weights_by_step_index,
                                    connection_expand_dimensions_by_step_index,
                                    constants_by_step_index, variables_by_step_index,
                                    time_and_variable_invariant_tensors_by_step_index,
                                    time_invariant_variable_variant_tensors_by_step_index):
    logger.info(f"Creating new unrolled simulation call with {num_time_steps} time steps per call")
    before = time.time()
    simulation_call = lambda start_time, values: simulate_unrolled_time_steps(num_time_steps, start_time,
                                                                              time_step_duration,
                                                                              steps,
                                                                              input_step_indices_by_step_index,
                                                                              activation_function_types_by_step_index,
                                                                              activation_function_betas_by_step_index,
                                                                              connection_kernel_weights_by_step_index,
                                                                              connection_pointwise_weights_by_step_index,
                                                                              connection_contract_dimensions_by_step_index,
                                                                              connection_contraction_weights_by_step_index,
                                                                              connection_expand_dimensions_by_step_index,
                                                                              constants_by_step_index,
                                                                              variables_by_step_index,
                                                                              time_and_variable_invariant_tensors_by_step_index,
                                                                              time_invariant_variable_variant_tensors_by_step_index,
                                                                              values)
    logger.debug("Done creating simulation call after " + str(time.time() - before) + " seconds")
    return simulation_call

def create_unrolled_simulation_call_with_history(num_time_steps,
                                    time_step_duration,
                                    steps,
                                    input_step_indices_by_step_index,
                                    activation_function_types_by_step_index,
                                    activation_function_betas_by_step_index,
                                    connection_kernel_weights_by_step_index,
                                    connection_pointwise_weights_by_step_index,
                                    connection_contract_dimensions_by_step_index,
                                    connection_contraction_weights_by_step_index,
                                    connection_expand_dimensions_by_step_index,
                                    constants_by_step_index, variables_by_step_index,
                                    time_and_variable_invariant_tensors_by_step_index,
                                    time_invariant_variable_variant_tensors_by_step_index):
    logger.info(f"Creating new unrolled simulation call with history with {num_time_steps} time steps per call")
    before = time.time()
    simulation_call = lambda start_time, values: simulate_unrolled_time_steps_with_history(num_time_steps, start_time,
                                                                              time_step_duration,
                                                                              steps,
                                                                              input_step_indices_by_step_index,
                                                                              activation_function_types_by_step_index,
                                                                              activation_function_betas_by_step_index,
                                                                              connection_kernel_weights_by_step_index,
                                                                              connection_pointwise_weights_by_step_index,
                                                                              connection_contract_dimensions_by_step_index,
                                                                              connection_contraction_weights_by_step_index,
                                                                              connection_expand_dimensions_by_step_index,
                                                                              constants_by_step_index,
                                                                              variables_by_step_index,
                                                                              time_and_variable_invariant_tensors_by_step_index,
                                                                              time_invariant_variable_variant_tensors_by_step_index,
                                                                              values)
    logger.debug("Done creating simulation call after " + str(time.time() - before) + " seconds")
    return simulation_call


def create_rolled_simulation_call(time_step_duration,
                                  steps,
                                  input_step_indices_by_step_index,
                                  activation_function_types_by_step_index,
                                  activation_function_betas_by_step_index,
                                  connection_kernel_weights_by_step_index,
                                  connection_pointwise_weights_by_step_index,
                                  connection_contract_dimensions_by_step_index,
                                  connection_contraction_weights_by_step_index,
                                  connection_expand_dimensions_by_step_index,
                                  constants_by_step_index, variables_by_step_index,
                                  time_and_variable_invariant_tensors_by_step_index,
                                  time_invariant_variable_variant_tensors_by_step_index):
    logger.info(f"Creating new rolled simulation call")
    before = time.time()
    simulation_call = lambda start_time, rolled_num_time_steps, values: simulate_rolled_time_steps(
        rolled_num_time_steps,
        start_time,
        time_step_duration,
        steps,
        input_step_indices_by_step_index,
        activation_function_types_by_step_index,
        activation_function_betas_by_step_index,
        connection_kernel_weights_by_step_index,
        connection_pointwise_weights_by_step_index,
        connection_contract_dimensions_by_step_index,
        connection_contraction_weights_by_step_index,
        connection_expand_dimensions_by_step_index,
        constants_by_step_index,
        variables_by_step_index,
        time_and_variable_invariant_tensors_by_step_index,
        time_invariant_variable_variant_tensors_by_step_index,
        values)
    logger.debug("Done creating simulation call after " + str(time.time() - before) + " seconds")
    return simulation_call


@tf.function
def simulate_unrolled_time_steps(num_time_steps, start_time, time_step_duration, steps, input_step_indices_by_step_index,
                                 activation_function_types_by_step_index, activation_function_betas_by_step_index,
                                 connection_kernel_weights_by_step_index, connection_pointwise_weights_by_step_index,
                                 connection_contract_dimensions_by_step_index, connection_contraction_weights_by_step_index,
                                 connection_expand_dimensions_by_step_index, constants_by_step_index,
                                 variables_by_step_index, time_and_variable_invariant_tensors_by_step_index,
                                 time_invariant_variable_variant_tensors_by_step_index,
                                 values):
    logger.debug(f"trace simulate_unrolled_time_steps")

    for time_step in range(num_time_steps):
        time_step_tensor = tf.constant(time_step)
        # TODO: Why does tracing here take twice as much time?
        simulate_time_step(time_step_tensor, start_time, time_step_duration, steps, input_step_indices_by_step_index,
                           activation_function_types_by_step_index, activation_function_betas_by_step_index,
                           connection_kernel_weights_by_step_index, connection_pointwise_weights_by_step_index,
                           connection_contract_dimensions_by_step_index, connection_contraction_weights_by_step_index,
                           connection_expand_dimensions_by_step_index,
                           constants_by_step_index, variables_by_step_index, time_and_variable_invariant_tensors_by_step_index,
                           time_invariant_variable_variant_tensors_by_step_index,
                           values)
    return values


@tf.function
def simulate_unrolled_time_steps_with_history(num_time_steps, start_time, time_step_duration, steps, input_step_indices_by_step_index,
                                 activation_function_types_by_step_index, activation_function_betas_by_step_index,
                                 connection_kernel_weights_by_step_index, connection_pointwise_weights_by_step_index,
                                 connection_contract_dimensions_by_step_index, connection_contraction_weights_by_step_index,
                                 connection_expand_dimensions_by_step_index, constants_by_step_index,
                                 variables_by_step_index, time_and_variable_invariant_tensors_by_step_index,
                                 time_invariant_variable_variant_tensors_by_step_index,
                                 values):
    logger.debug(f"trace simulate_unrolled_time_steps")

    history = [values]
    for time_step in range(num_time_steps):
        time_step_tensor = tf.constant(time_step)
        # TODO: Why does tracing here take twice as much time?
        new_values = simulate_time_step(time_step_tensor, start_time, time_step_duration, steps, input_step_indices_by_step_index,
                           activation_function_types_by_step_index, activation_function_betas_by_step_index,
                           connection_kernel_weights_by_step_index, connection_pointwise_weights_by_step_index,
                           connection_contract_dimensions_by_step_index, connection_contraction_weights_by_step_index,
                           connection_expand_dimensions_by_step_index,
                           constants_by_step_index, variables_by_step_index, time_and_variable_invariant_tensors_by_step_index,
                           time_invariant_variable_variant_tensors_by_step_index,
                           values)
        history.append(new_values)
    return history


@tf.function
def simulate_rolled_time_steps(num_time_steps, start_time, time_step_duration, steps, input_step_indices_by_step_index,
                               activation_function_types_by_step_index, activation_function_betas_by_step_index,
                               connection_kernel_weights_by_step_index, connection_pointwise_weights_by_step_index,
                               connection_contract_dimensions_by_step_index, connection_contraction_weights_by_step_index,
                               connection_expand_dimensions_by_step_index,
                               constants_by_step_index, variables_by_step_index,
                               time_and_variable_invariant_tensors_by_step_index,
                               time_invariant_variable_variant_tensors_by_step_index,
                               values):
    logger.debug("trace simulate_rolled_time_steps") #, time, num_time_steps, time_step_duration, steps, connections_into_steps, step_constants, step_variables, time_and_variable_invariant_tensors_by_step_index, values

    for time_step in tf.range(num_time_steps):
        simulate_time_step(time_step, start_time, time_step_duration, steps, input_step_indices_by_step_index,
                           activation_function_types_by_step_index, activation_function_betas_by_step_index,
                           connection_kernel_weights_by_step_index, connection_pointwise_weights_by_step_index,
                           connection_contract_dimensions_by_step_index, connection_contraction_weights_by_step_index,
                           connection_expand_dimensions_by_step_index,
                           constants_by_step_index, variables_by_step_index, time_and_variable_invariant_tensors_by_step_index,
                           time_invariant_variable_variant_tensors_by_step_index, values)
    return values


@tf.function
def simulate_time_step(time_step, start_time, time_step_duration, steps, input_step_indices_by_step_index,
                       activation_function_types_by_step_index, activation_function_betas_by_step_index,
                       connection_kernel_weights_by_step_index, connection_pointwise_weights_by_step_index,
                       connection_contract_dimensions_by_step_index, connection_contraction_weights_by_step_index,
                       connection_expand_dimensions_by_step_index,
                       step_constants,
                       variables_by_step_index, time_and_variable_invariant_tensors_by_step_index,
                       time_invariant_variable_variant_tensors_by_step_index, current_values):
    logger.debug(f"trace simulate_time_step")
    #before = time.time()
    # TODO see if performance can be improved by not creating a copy here
    # e.g., just an empty list, with or without specification of size, content shapes, ...
    new_values = current_values.copy()
    for i in range(0, len(steps)):
        step = steps[i]
        if not step.static:
            constants = step_constants[i]
            variables = variables_by_step_index[i]
            time_and_variable_invariant_tensors = time_and_variable_invariant_tensors_by_step_index[i]
            time_invariant_variable_variant_tensors = time_invariant_variable_variant_tensors_by_step_index[i]
            input_steps_indices = input_step_indices_by_step_index[i]
            input_steps_activation_function_types = activation_function_types_by_step_index[i]
            input_steps_activation_function_betas = activation_function_betas_by_step_index[i]
            input_steps_connection_kernel_weights = connection_kernel_weights_by_step_index[i]
            input_steps_connection_pointwise_weights = connection_pointwise_weights_by_step_index[i]
            input_steps_contract_dimensions = connection_contract_dimensions_by_step_index[i]
            input_steps_contraction_weights = connection_contraction_weights_by_step_index[i]
            input_steps_expand_dimensions = connection_expand_dimensions_by_step_index[i]
            step_shape = tf.shape(new_values[i])

            if len(input_steps_indices) == 0:
                input_sum = tf.zeros(shape=step_shape)
            else:
                input_steps_current_values = get_input_steps_current_values(input_steps_indices, current_values)
                input_sum = get_input_sum(input_steps_activation_function_types, input_steps_activation_function_betas,
                                          input_steps_connection_kernel_weights,
                                          input_steps_connection_pointwise_weights, input_steps_contract_dimensions,
                                          input_steps_contraction_weights,
                                          input_steps_expand_dimensions,
                                          input_steps_current_values, step_shape)


            if isinstance(step, TimedBoost):
                new_values[i] = dff.simulation.steps.timed_boost.timed_boost_time_step(constants[0],
                                                                                       start_time + time_step_duration
                                                                                       * tf.cast(time_step, tf.float32))
            elif isinstance(step, TimedGate):
                new_values[i] = dff.simulation.steps.timed_gate.timed_gate_time_step(constants[0], constants[1],
                                                                                     input_sum,
                                                                                     start_time + time_step_duration
                                                                                       * tf.cast(time_step, tf.float32))
            elif isinstance(step, Boost):
                new_values[i] = dff.simulation.steps.boost.boost_time_step(constants[0])
            elif isinstance(step, Field):
                resting_level_tensor = time_invariant_variable_variant_tensors[0]
                lateral_interaction_weight_pattern_tensor = time_invariant_variable_variant_tensors[1]
                new_values[i] = dff.simulation.steps.field.field_time_step(time_step_duration,
                                                                           constants[1],
                                                                           constants[2],
                                                                           variables[1],
                                                                           variables[2],
                                                                           variables[3],
                                                                           variables[4],
                                                                           resting_level_tensor,
                                                                           lateral_interaction_weight_pattern_tensor,
                                                                           input_sum,
                                                                           current_values[i])
            elif isinstance(step, Node):
                new_values[i] = dff.simulation.steps.node.node_time_step(time_step_duration,
                                                                         variables[0], variables[1],
                                                                         variables[2], variables[3],
                                                                         variables[4],
                                                                         input_sum, current_values[i])
            elif isinstance(step, ScalarMultiplication):
                new_values[i] = dff.simulation.steps.scalar_multiplication.scalar_multiplication_time_step(
                    variables[0],
                    input_sum)
            elif isinstance(step, NoiseInput):
                new_values[i] = dff.simulation.steps.noise_input.noise_input_time_step(time_step_duration, constants[0],
                                                                                       constants[1])

    for i in range(0, len(steps)):
        current_values[i].assign(new_values[i])

    return new_values


#@tf.function
def compute_output(input, activation_function_type, beta):
    logger.debug(f"trace apply_projection {input} {beta}")

    if activation_function_type == 1:
        input = tf.math.sigmoid(tf.multiply(beta, input))

    return input


@tf.function
def get_input_sum(input_steps_activation_function_types, input_steps_activation_function_betas,
                  input_steps_connection_kernel_weights, input_steps_connection_pointwise_weights,
                  input_steps_contract_dimensions, input_steps_contraction_weights, input_steps_expand_dimensions,
                  input_steps_values, step_shape):
    logger.debug(f"trace get_input_sum {input_steps_activation_function_betas} {input_steps_connection_kernel_weights} {input_steps_connection_pointwise_weights} {input_steps_contract_dimensions} {input_steps_contraction_weights} {input_steps_expand_dimensions} {input_steps_values} {step_shape}")

    # Handle the first incoming connection
    activation_function_type = input_steps_activation_function_types[0]
    beta = input_steps_activation_function_betas[0]
    connection_kernel_weights = input_steps_connection_kernel_weights[0]
    connection_pointwise_weights = input_steps_connection_pointwise_weights[0]
    connection_contract_dimensions = input_steps_contract_dimensions[0]
    connection_contraction_weights = input_steps_contraction_weights[0]
    connection_expand_dimensions = input_steps_expand_dimensions[0]

    input = input_steps_values[0]
    if tf.rank(input) == 0:
        input = tf.ones(step_shape) * input

    input = compute_output(input, activation_function_type, beta)
    if connection_contract_dimensions is not None and len(connection_contract_dimensions) > 0:
        input = contract(input, connection_contract_dimensions, connection_contraction_weights)
    elif connection_expand_dimensions is not None and len(connection_expand_dimensions) > 0:
        input = expand(input, connection_expand_dimensions, step_shape)
    if connection_pointwise_weights is not None:
        input = tf.math.multiply(connection_pointwise_weights, input)
    if connection_kernel_weights is not None:
        input = convolve(connection_kernel_weights, input)

    input_sum = input

    # Iterate all other incoming connections
    for j in range(1, len(input_steps_values)):
        input = input_steps_values[j]
        activation_function_type = input_steps_activation_function_types[j]
        beta = input_steps_activation_function_betas[j]
        connection_kernel_weights = input_steps_connection_kernel_weights[j]
        connection_pointwise_weights = input_steps_connection_pointwise_weights[j]
        connection_contract_dimensions = input_steps_contract_dimensions[j]
        connection_contraction_weights = input_steps_contraction_weights[j]
        connection_expand_dimensions = input_steps_expand_dimensions[j]

        input = compute_output(input, activation_function_type, beta)
        if len(input.shape) > len(step_shape) and connection_expand_dimensions is not None:
            input = contract(input, connection_contract_dimensions, connection_contraction_weights)
        elif len(input.shape) < len(step_shape) and connection_expand_dimensions is not None:
            input = expand(input, connection_expand_dimensions, step_shape)
        if connection_pointwise_weights is not None:
            input = tf.math.multiply(connection_pointwise_weights, input)
        if connection_kernel_weights is not None:
            input = convolve(connection_kernel_weights, input)

        input_sum = tf.add(input_sum, input)

    return input_sum


#@tf.function
def contract(input, contract_dimensions, contraction_weights):
    logger.debug(f"trace contract {input} {contract_dimensions} {contraction_weights}")
    if contraction_weights is not None:
        input = tf.multiply(input, contraction_weights)

    for i in range(len(contract_dimensions)):
        input = tf.reduce_sum(input, axis=contract_dimensions[i])
    return input


#@tf.function
def expand(input, expand_dimensions, output_shape):
    logger.debug(f"trace expand {input} {expand_dimensions} {output_shape}")
    for i in range(len(expand_dimensions)):
        dim = expand_dimensions[i]
        dimension_length = output_shape[dim]
        input_shape = tf.shape(input)
        input = tf.repeat(input, [dimension_length])
        new_shape = tf.concat([input_shape, [dimension_length]], axis=0)
        input = tf.reshape(input, new_shape)
    return input


#get_input_sum_1d = tf.function(input_signature=(
#    tf.TensorSpec(shape=(None,), dtype=tf.float32),
#    tf.TensorSpec(shape=(None,), dtype=tf.float32),
#    tf.TensorSpec(shape=(None,None), dtype=tf.float32),
#))(get_input_sum)


def get_input_steps_current_values(input_step_indices, current_values):
    input_steps_current_values = []
    for i in range(len(input_step_indices)):
        input_steps_current_values.append(current_values[input_step_indices[i]])
    return input_steps_current_values
