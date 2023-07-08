import time
import logging

from dff.simulation.convolution import convolve
from dff.simulation import steps
from dff.simulation.weight_patterns import compute_weight_pattern_tensor
from dfpy import Sigmoid
from dfpy.connection import SynapticConnection

logger = logging.getLogger(__name__)

import tensorflow as tf
from dfpy.steps import *
import dff.simulation.steps
import time

def create_unrolled_simulation_call(simulator, num_time_steps, time_step_duration):
    logger.info(f"Creating new unrolled simulation call with {num_time_steps} time steps per call")
    before = time.time()
    simulation_call = lambda start_time, values: simulate_unrolled_time_steps(simulator, num_time_steps, start_time,
                                                                              time_step_duration,
                                                                              values)
    logger.debug("Done creating simulation call after " + str(time.time() - before) + " seconds")
    return simulation_call

def create_unrolled_simulation_call_with_history(simulator, num_time_steps, time_step_duration):
    logger.info(f"Creating new unrolled simulation call with history with {num_time_steps} time steps per call")
    before = time.time()
    simulation_call = lambda start_time, values: simulate_unrolled_time_steps_with_history(simulator, num_time_steps, start_time,
                                                                              time_step_duration,
                                                                              values)
    logger.debug("Done creating simulation call after " + str(time.time() - before) + " seconds")
    return simulation_call

def create_impromptu_simulation_call_with_history(simulator, num_time_steps, time_step_duration):
    logger.info(f"Creating new unrolled simulation call with history with {num_time_steps} time steps per call")
    before = time.time()
    simulation_call = lambda start_time, values:\
        simulate_impromptu_time_steps_with_history(simulator, num_time_steps, start_time, time_step_duration, values)
    logger.debug("Done creating simulation call after " + str(time.time() - before) + " seconds")
    return simulation_call


def create_rolled_simulation_call(simulator, time_step_duration):
    logger.info(f"Creating new rolled simulation call")
    before = time.time()
    simulation_call = lambda start_time, rolled_num_time_steps, values: simulate_rolled_time_steps(
        simulator,
        rolled_num_time_steps,
        start_time,
        time_step_duration,
        values)
    logger.debug("Done creating simulation call after " + str(time.time() - before) + " seconds")
    return simulation_call


@tf.function
def simulate_unrolled_time_steps(simulator, num_time_steps, start_time, time_step_duration, values):
    logger.debug(f"trace simulate_unrolled_time_steps")

    for relative_time_step in range(num_time_steps):
        time_step = tf.cast(relative_time_step + start_time/time_step_duration, tf.int32)
        # TODO: Why does tracing here take twice as much time?
        new_values = simulate_time_step(simulator, time_step, start_time, time_step_duration, values)

        for i in range(0, len(simulator._neural_structure.steps)):
            values[i].assign(new_values[i])

    return values

@tf.function
def simulate_unrolled_time_steps_with_history(simulator, num_time_steps, start_time, time_step_duration, values):
    logger.debug(f"trace simulate_unrolled_time_steps")


    # Prepare cache of time-invariant variable-variant tensors
    for i in range(0, len(simulator._neural_structure.steps)):
        step = simulator._neural_structure.steps[i]
        if not step.static and step.trainable:
            constants = simulator._constants[step]
            variables = simulator._variables[step]
            time_invariant_variable_variant_tensors = simulator._time_invariant_variable_variant_tensors[step]
            time_and_variable_invariant_tensors = simulator._time_and_variable_invariant_tensors[step]
            resting_level_tensor = tf.ones(tuple([int(x) for x in constants["shape"]])) * variables["resting_level"]
            lateral_interaction_weight_pattern_tensor = compute_weight_pattern_tensor(variables["interaction_kernel_weight_pattern_config"],
                                                                                      time_and_variable_invariant_tensors["interaction_kernel_positional_grid"])
            time_invariant_variable_variant_tensors[step] = {}
            time_invariant_variable_variant_tensors[step]["resting_level_tensor"] = resting_level_tensor
            time_invariant_variable_variant_tensors[step]["lateral_interaction_weight_pattern_tensor"] = lateral_interaction_weight_pattern_tensor


    history = [values]
    for time_step in range(num_time_steps):
        time_step_tensor = tf.constant(time_step)
        # TODO: Why does tracing here take twice as much time?
        values = simulate_time_step(simulator, time_step_tensor, start_time, time_step_duration, values)
        history.append(values)
    return history


def simulate_impromptu_time_steps_with_history(simulator, num_time_steps, start_time, time_step_duration, values):
    logger.debug(f"trace simulate_unrolled_time_steps")
    history = [values]
    for time_step in range(num_time_steps):
        time_step_tensor = tf.constant(time_step)
        # TODO: Why does tracing here take twice as much time?
        values = simulate_time_step(simulator, time_step_tensor, start_time, time_step_duration, values)
        history.append(values)
    return history



@tf.function
def simulate_rolled_time_steps(simulator, num_time_steps, start_time, time_step_duration, values):
    logger.debug("trace simulate_rolled_time_steps")
    for time_step in tf.range(num_time_steps):
        values = simulate_time_step(simulator, time_step, start_time, time_step_duration, values)
    return values


@tf.function
def simulate_time_step(simulator, time_step, start_time, time_step_duration, current_values):
    #logger.debug(f"trace simulate_time_step")

    #before = time.time()
    # TODO see if performance can be improved by not creating a copy here
    # e.g., just an empty list, with or without specification of size, content shapes, ...
    new_values = current_values.copy()
    for i in range(0, len(simulator._neural_structure.steps)):
        step = simulator._neural_structure.steps[i]
        if not step.static:
            constants = simulator._constants[step]
            variables = simulator._variables[step]
            time_and_variable_invariant_tensors = simulator._time_and_variable_invariant_tensors[step]
            time_invariant_variable_variant_tensors = simulator._time_invariant_variable_variant_tensors[step]
            step_shape = tf.shape(new_values[i])

            if len(simulator._neural_structure.connections_into_steps[i]) == 0:
                input_sum = tf.zeros(shape=step_shape)
            else:
                input_steps_current_values = get_input_steps_current_values(simulator._neural_structure.connections_into_steps[i], current_values)
                input_sum = get_input_sum(simulator, input_steps_current_values, step_shape, i)

            if isinstance(step, TimedBoost):
                new_values[i] = dff.simulation.steps.timed_boost.timed_boost_time_step(constants["values"],
                                                                                       start_time + time_step_duration
                                                                                       * tf.cast(time_step, tf.float32))
            elif isinstance(step, TimedGate):
                new_values[i] = dff.simulation.steps.timed_gate.timed_gate_time_step(constants["min_time"], constants["max_time"],
                                                                                     input_sum,
                                                                                     start_time + time_step_duration
                                                                                       * tf.cast(time_step, tf.float32))
            elif isinstance(step, TimedCustomInput):
                new_values[i] = dff.simulation.steps.timed_custom_input.timed_custom_input_time_step(variables["timed_custom_input"], time_step)
            elif isinstance(step, Boost):
                new_values[i] = dff.simulation.steps.boost.boost_time_step(constants["value"])
            elif isinstance(step, Field):
                if not step.trainable:
                    resting_level_tensor = time_invariant_variable_variant_tensors["resting_level_tensor"]
                    lateral_interaction_weight_pattern_tensor = time_invariant_variable_variant_tensors["lateral_interaction_weight_pattern_tensor"]
                else:
                    resting_level_tensor = time_invariant_variable_variant_tensors[step]["resting_level_tensor"]
                    lateral_interaction_weight_pattern_tensor = time_invariant_variable_variant_tensors[step]["lateral_interaction_weight_pattern_tensor"]

                new_values[i] = dff.simulation.steps.field.field_time_step(time_step_duration,
                                                                           constants["shape"],
                                                                           constants["bin_size"],
                                                                           variables["time_scale"],
                                                                           variables["sigmoid_beta"],
                                                                           variables["global_inhibition"],
                                                                           variables["noise_strength"],
                                                                           resting_level_tensor,
                                                                           lateral_interaction_weight_pattern_tensor,
                                                                           input_sum,
                                                                           current_values[i])

            elif isinstance(step, Node):
                new_values[i] = dff.simulation.steps.node.node_time_step(time_step_duration,
                                                                         variables["resting_level"], variables["time_scale"],
                                                                         variables["sigmoid_beta"], variables["self_excitation"],
                                                                         variables["noise_strength"],
                                                                         input_sum, current_values[i])
            elif isinstance(step, ScalarMultiplication):
                new_values[i] = dff.simulation.steps.scalar_multiplication.scalar_multiplication_time_step(
                    variables[0],
                    input_sum)
            elif isinstance(step, NoiseInput):
                new_values[i] = dff.simulation.steps.noise_input.noise_input_time_step(time_step_duration, step.shape,
                                                                                       step.strength)
            elif isinstance(step, GaussInput):
                new_values[i] = dff.simulation.steps.gauss_input.gauss_input_time_step(variables["height"],
                                                                                       variables["mean"],
                                                                                       variables["sigmas"],
                                                                                       time_and_variable_invariant_tensors["positional_grid"])

    return new_values


#@tf.function
def get_input_sum(simulator, input_steps_values, step_shape, i):
    #logger.debug(f"trace get_input_sum")
    step = simulator._neural_structure.steps[i]

    if i == 0:
        before = time.time()
        before_all = time.time()

    connections = simulator._neural_structure.connections_into_steps[i]
    contraction_weights_by_connection = simulator._connection_contraction_weights[step]
    kernel_weights_by_connection = simulator._connection_kernel_weights[step]
    kernel_weight_pattern_configs_by_connection = simulator._connection_kernel_weight_pattern_configs[step]
    kernel_positional_grids_by_connection = simulator._connection_kernel_positional_grids[step]
    pointwise_weights_by_connection = simulator._connection_pointwise_weights[step]


    # Handle the first incoming connection
    input = input_steps_values[0]

    if len(input.shape) == 0:
        input = tf.ones(step_shape) * input

    if isinstance(connections[0], SynapticConnection):
        if isinstance(connections[0].activation_function, Sigmoid):
            input = tf.math.sigmoid(tf.multiply(connections[0].activation_function.beta, input))

    if connections[0].contract_dimensions is not None and len(connections[0].contract_dimensions) > 0:

        # Contract
        if contraction_weights_by_connection[0] is not None:
            input = tf.multiply(input, contraction_weights_by_connection[0])
        for i in range(len(connections[0].contract_dimensions)):
            input = tf.reduce_sum(input, axis=connections[0].contract_dimensions[i])

    elif connections[0].expand_dimensions is not None and len(connections[0].expand_dimensions) > 0:

        # Expand
        for i in range(len(connections[0].expand_dimensions)):
            dim = connections[0].expand_dimensions[i]
            dimension_length = step_shape[dim]
            input_shape = tf.shape(input)
            input = tf.repeat(input, [dimension_length])
            new_shape = tf.concat([input_shape, [dimension_length]], axis=0)
            input = tf.reshape(input, new_shape)

    if pointwise_weights_by_connection[0] is not None:
        input = tf.math.multiply(pointwise_weights_by_connection[0], input)

    # TODO Do this before the first time step
    if connections[0].trainable:
        if kernel_weight_pattern_configs_by_connection[0] is not None:
            kernel_weights = compute_weight_pattern_tensor(kernel_weight_pattern_configs_by_connection[0], kernel_positional_grids_by_connection[0])
            input = convolve(input, kernel_weights)
    else:
        if kernel_weights_by_connection[0] is not None:
            input = convolve(input, kernel_weights_by_connection[0])

    input_sum = input

    # Iterate all other incoming connections
    for j in range(1, len(input_steps_values)):

        input = input_steps_values[j]

        if isinstance(connections[j], SynapticConnection):
            if isinstance(connections[j].activation_function, Sigmoid):
                input = tf.math.sigmoid(tf.multiply(connections[j].activation_function.beta, input))

        if connections[j].contract_dimensions is not None and len(connections[j].contract_dimensions) > 0:

            # Contract
            if contraction_weights_by_connection[j] is not None:
                input = tf.multiply(input, contraction_weights_by_connection[j])
            for i in range(len(connections[j].contract_dimensions)):
                input = tf.reduce_sum(input, axis=connections[j].contract_dimensions[i])

        elif connections[j].expand_dimensions is not None and len(connections[j].expand_dimensions) > 0:

            # Expand
            for i in range(len(connections[j].expand_dimensions)):
                dim = connections[j].expand_dimensions[i]
                dimension_length = step_shape[dim]
                input_shape = tf.shape(input)
                input = tf.repeat(input, [dimension_length])
                new_shape = tf.concat([input_shape, [dimension_length]], axis=0)
                input = tf.reshape(input, new_shape)

        if pointwise_weights_by_connection[j] is not None:
            input = tf.math.multiply(pointwise_weights_by_connection[j], input)

        if connections[j].trainable:
            if kernel_weight_pattern_configs_by_connection[j] is not None:
                # TODO do this before the first time step
                kernel_weights = compute_weight_pattern_tensor(kernel_weight_pattern_configs_by_connection[j], kernel_positional_grids_by_connection[j])
                input = convolve(input, kernel_weights)
        else:
            if kernel_weights_by_connection[j] is not None:
                input = convolve(input, kernel_weights_by_connection[j])

        input_sum = tf.add(input_sum, input)

    return input_sum


def get_input_steps_current_values(connections_into_steps, current_values):
    input_steps_current_values = []
    for i in range(len(connections_into_steps)):
        connection = connections_into_steps[i]
        input_step_index = connection.input_step_index
        input_steps_current_values.append(current_values[input_step_index])
    return input_steps_current_values
