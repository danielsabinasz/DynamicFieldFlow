import tensorflow as tf
import logging


def timed_custom_input_prepare_variables(step):
    if step.assignable:
        timed_custom_input = tf.Variable(step.timed_custom_input, dtype=tf.float32, trainable=step.trainable)
    else:
        timed_custom_input = tf.constant(step.timed_custom_input, dtype=tf.float32)
    return {"timed_custom_input": timed_custom_input}


@tf.function
def timed_custom_input_time_step(timed_custom_input, time_step):
    logging.debug(f"trace timed_custom_input_time_step {timed_custom_input} {time_step}")
    return timed_custom_input[time_step]
