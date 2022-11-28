import tensorflow as tf


def custom_input_prepare_variables(step):
    pattern = tf.Variable(step.pattern, dtype=tf.float32)
    return {"pattern": pattern}
