import tensorflow as tf


def custom_input_prepare_variables(step):
    if step.assignable:
        pattern = tf.Variable(step.pattern, dtype=tf.float32, trainable=step.trainable)
    else:
        pattern = tf.constant(step.pattern)
    return {"pattern": pattern}
