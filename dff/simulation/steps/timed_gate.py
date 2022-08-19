import logging

import tensorflow as tf


def timed_gate_prepare_constants(step):
    min_time = tf.convert_to_tensor(step.min_time)
    max_time = tf.convert_to_tensor(step.max_time)
    return {"min_time": min_time, "max_time": max_time}


@tf.function
def timed_gate_time_step(min_time, max_time, input, time):
    logging.debug("trace timed_gate_time_step")
    if time >= min_time and time <= max_time:
        return input
    else:
        return tf.zeros(input.shape)

