import tensorflow as tf


def scalar_prepare_variables(step):
    if step.assignable:
        value = tf.Variable(step.value, trainable=step.trainable)
    else:
        value = tf.constant(step.value, trainable=step.trainable)
    return {"value": value}
