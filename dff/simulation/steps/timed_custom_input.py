import tensorflow as tf
import logging


def timed_custom_input_prepare_variables(step):
    timed_custom_input = tf.Variable(step.timed_custom_input, dtype=tf.float32)
    return {"timed_custom_input": timed_custom_input}


@tf.function
def timed_custom_input_time_step(timed_custom_input, time_step):
    logging.debug("trace timed_custom_input_time_step")
    return timed_custom_input[time_step]
