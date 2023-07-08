import logging

import tensorflow as tf


def node_prepare_constants(step):
    return {}


def node_prepare_variables(step):
    if step.assignable:
        resting_level = tf.Variable(step.resting_level, trainable=step.trainable)
        time_scale = tf.Variable(step.time_scale, trainable=step.trainable)
        sigmoid_beta = tf.Variable(step.activation_function.beta, trainable=step.trainable)
        self_excitation = tf.Variable(step.self_excitation, trainable=step.trainable)
        noise_strength = tf.Variable(step.noise_strength, trainable=step.trainable)
    else:
        resting_level = tf.constant(step.resting_level)
        time_scale = tf.constant(step.time_scale)
        sigmoid_beta = tf.constant(step.activation_function.beta)
        self_excitation = tf.constant(step.self_excitation)
        noise_strength = tf.constant(step.noise_strength)
    return {"resting_level": resting_level, "time_scale": time_scale,
            "sigmoid_beta": sigmoid_beta,
            "self_excitation": self_excitation, "noise_strength": noise_strength}


@tf.function
def node_time_step(time_step_duration, resting_level, time_scale, self_excitation_sigmoid_beta, self_excitation,
                   noise_strength,
                   input=None, activation=None):
    """Computes a time step of the Node step.

    :param Tensor time_step_duration: duration of a time step in milliseconds
    :param Tensor resting_level: resting level of the node activation
    :param Tensor time_scale: time scale of the node (parameter tau in the dynamics)
    :param Tensor self_excitation_sigmoid_beta: beta parameter of the sigmoid (slope)
    :param Tensor self_excitation: self-excitation of the node
    :param Tensor input: input to the node
    :param Tensor activation: node activation from the previous time step
    :return: Tensor: node activation
    """
    logging.debug("trace node_time_step")

    if activation is None:
        activation = resting_level
    if input == None:
        input = tf.constant(0.0)

    minus_u = tf.multiply(-1.0, activation)
    output = tf.math.sigmoid(tf.multiply(self_excitation_sigmoid_beta, activation))
    conv_result = output * self_excitation
    noise_term = tf.multiply(tf.sqrt(time_step_duration), noise_strength)
    noise_term = tf.multiply(noise_term, tf.random.normal(()))
    sum = tf.add_n([minus_u, resting_level, input, conv_result, noise_term])

    # Compute a rate of change
    rate_of_change = tf.divide(sum, time_scale)

    # Compute the absolute change to the stored activation (considering discretized time)
    absolute_change = tf.multiply(time_step_duration, rate_of_change)

    activation = tf.add(activation, absolute_change)

    return activation
