import logging

import tensorflow as tf


def noise_input_prepare_constants(step):
    return {}


@tf.function
def noise_input_time_step(time_step_duration, shape, strength):
    """Computes a time step of the NoiseInput step.

    :param Tensor time_step_duration: duration of a time step in milliseconds
    :param Tensor shape: shape of the field
    :param Tensor strength: noise strength (multiplier)
    :param Tensor time_scale: time scale of the field (parameter tau in the field dynamics)
    :return: Tensor: noise input
    """
    logging.debug(f"trace noise_input_time_step: time_step_duration={time_step_duration}, shape={shape}, strength={strength}")
    return 1. / tf.sqrt(time_step_duration) * tf.random.normal(shape, 0, strength)
