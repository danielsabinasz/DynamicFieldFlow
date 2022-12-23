import PIL.Image
import tensorflow as tf
from dff.simulation.weight_patterns import compute_kernel_gauss_tensor

from dff.simulation.convolution import convolve


def create_hue_space_perception_field_input(field, filename):
    shape = field.shape()
    domain = field.domain()

    number_of_bins = shape[2]

    image = PIL.Image.open(filename)
    image = image.resize((shape[0], shape[1]))
    image = image.convert("HSV")

    image_as_array = tf.keras.utils.img_to_array(image)
    hue_map = image_as_array[:,:,0]
    saturation_map = image_as_array[:,:,1]

    lower_limit = 0.0
    upper_limit = 255.0

    # Compute a binning of the interval [lower_limit, upper_limit]
    bins_lower_bounds = lower_limit\
                        + tf.range(number_of_bins, dtype=tf.float32) / number_of_bins\
                        * (upper_limit - lower_limit)
    bins_upper_bounds = lower_limit\
                        + tf.range(1, number_of_bins+1, dtype=tf.float32) / number_of_bins\
                        * (upper_limit - lower_limit)

    width = shape[0]
    height = shape[1]

    # Expand bin map by another axis of length number_of_bins
    bin_map_expanded = tf.reshape(tf.repeat(hue_map, number_of_bins), (width, height, number_of_bins))

    # Expand lower bounds by two axes of length width and height, respectively
    bins_lower_bounds_expanded = tf.transpose(
        tf.reshape(
            tf.repeat(bins_lower_bounds, width*height),
            (number_of_bins, width, height)
        ), perm=[1, 2, 0])

    # Expand upper bounds by two axes of length width and height, respectively
    bins_upper_bounds_expanded = tf.transpose(
        tf.reshape(
            tf.repeat(bins_upper_bounds, width*height),
            (number_of_bins, width, height)
        ), perm=[1, 2, 0])

    mask = tf.math.logical_and(
        tf.math.greater_equal(bin_map_expanded, bins_lower_bounds_expanded),
        tf.math.less(bin_map_expanded, bins_upper_bounds_expanded)
    )

    output = tf.reshape(tf.repeat(saturation_map, number_of_bins), (width, height, number_of_bins)) * tf.cast(mask, tf.float32)

    return output/255.0
