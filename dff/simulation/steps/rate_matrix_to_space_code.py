import logging

import tensorflow as tf


def neural_field_prepare_constants(step):
    number_of_bins = tf.convert_to_tensor(step.number_of_bins)
    lower_limit = tf.convert_to_tensor(step.lower_limit)
    upper_limit = tf.convert_to_tensor(step.upper_limit)
    bin_map = None
    return {"number_of_bins": number_of_bins, "lower_limit": lower_limit, "upper_limit": upper_limit,
            "bin_map": bin_map}


@tf.function
def rate_matrix_to_space_code(number_of_bins, lower_limit, upper_limit, bin_map, values=None):
    """Prepares constants for the RateMatrixToSpaceCode step.

    :param Tensor number_of_bins: number of bins
    :param Tensor lower_limit: lower limit of the values in the map
    :param Tensor upper_limit: upper limit of the values in the map
    :param Tensor bin_map: bin map
    :param Tensor values: values
    :return: Tensor: output tensor T
    """
    logging.debug("trace rate_matrix_to_space_code")

    # Compute a binning of the interval [lower_limit, upper_limit]
    bins_lower_bounds = lower_limit\
                        + tf.range(number_of_bins, dtype=tf.float32) / number_of_bins\
                        * (upper_limit - lower_limit)
    bins_upper_bounds = lower_limit\
                        + tf.range(1, number_of_bins+1, dtype=tf.float32) / number_of_bins\
                        * (upper_limit - lower_limit)

    width = bin_map.shape[0]
    height = bin_map.shape[1]

    # Expand bin map by another axis of length number_of_bins
    bin_map_expanded = tf.reshape(tf.repeat(bin_map, number_of_bins), (width, height, number_of_bins))

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

    output = tf.math.logical_and(
        tf.math.greater_equal(bin_map_expanded, bins_lower_bounds_expanded),
        tf.math.less(bin_map_expanded, bins_upper_bounds_expanded)
    )

    return output
