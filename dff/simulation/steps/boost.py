import logging

import tensorflow as tf


def boost_prepare_constants(step):
    value = tf.convert_to_tensor(step.value)
    return {"value": value}


@tf.function
def boost_time_step(value):
    logging.debug("trace boost_time_step")
    return value
