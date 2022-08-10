import tensorflow as tf


def custom_input_prepare_constants(step):
    pattern = tf.convert_to_tensor(step.pattern, dtype=tf.float32)
    return {"pattern": pattern}
