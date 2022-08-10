import logging

import tensorflow as tf


def timed_boost_prepare_constants(step):
    values = tf.convert_to_tensor(list(step.values.items()))
    return {"values": values}


@tf.function
def timed_boost_time_step(values, time):
    logging.debug("trace timed_boost_time_step")
    ret = tf.constant(0.0)
    for i in range(0, len(values) - 1):
        if values[i][0] <= time and time < values[i + 1][0]:
            ret = values[i][1]
    return ret
