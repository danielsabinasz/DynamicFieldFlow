import tensorflow as tf


def image_prepare_constants(step):
    image_tensor = tf.keras.preprocessing.image.img_to_array(step.image, dtype=tf.int32)

    return {"image_tensor": image_tensor}
