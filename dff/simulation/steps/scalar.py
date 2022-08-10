import tensorflow as tf


def scalar_prepare_variables(step):
    value = tf.Variable(step.value)
    return {"value": value}
