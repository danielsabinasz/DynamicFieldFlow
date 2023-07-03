import tensorflow as tf
import tensorflow_probability as tfp

from dff.simulation.util import compute_positional_grid
from dfpy import SumWeightPattern, GaussWeightPattern, RepeatedValueWeightPattern, CustomWeightPattern


def compute_weight_pattern_tensor(weight_pattern_config, positional_grid):
    """Creates a Tensor from a dff.weight_patterns configuration.

    :param dict weight_pattern_config: The weight pattern kernel configuration
    :param Tensor positional_grid: positional grid
    :return Tensor: The interaction kernel tensor
    """

    if weight_pattern_config["type"] == "GaussWeightPattern":
        result = compute_kernel_gauss_tensor_with_positional_grid(weight_pattern_config["mean"],
                                                                  weight_pattern_config["sigmas"],
                                                                  weight_pattern_config["height"],
                                                                  positional_grid)
    elif weight_pattern_config["type"] == "SumWeightPattern":
        summands = weight_pattern_config["summands"]
        tensors = [compute_weight_pattern_tensor(summand, positional_grid) for summand in summands]
        result = tf.add_n(tensors)
    #elif weight_pattern_config["type"] == "RepeatWeightPattern":
    #    inner_weight_pattern_config = weight_pattern_config["weight_pattern"]
    #    inner_weight_pattern_tensor = compute_weight_pattern_tensor(inner_weight_pattern_config, positional_grid[:,:,0,0:2])
    #    num_repeats = weight_pattern_config["num_repeats"]
    #    result = tf.repeat(tf.expand_dims(inner_weight_pattern_tensor, -1), num_repeats, axis=-1)
    elif weight_pattern_config["type"] == "RepeatedValueWeightPattern":
        result = tf.ones(weight_pattern_config["shape"])*weight_pattern_config["value"]
    elif weight_pattern_config["type"] == "CustomWeightPattern":
        result = weight_pattern_config["pattern"]
    elif weight_pattern_config["type"] == "ScalarWeightPattern":
        result = weight_pattern_config["scalar"]
    elif weight_pattern_config["type"] == "None" or  weight_pattern_config["type"] == None:
        result = tf.zeros(weight_pattern_config["shape"])
    else:
        raise RuntimeError("Unrecognized weight pattern type: " + str(weight_pattern_config["type"]))

    return result


def weight_pattern_config_from_dfpy_weight_pattern(dfpy_weight_pattern, domain, shape, previous_summand_sigmas=None):
    weight_pattern_config = {
        "shape": shape,
        "domain": domain
    }

    if isinstance(dfpy_weight_pattern, GaussWeightPattern):
        weight_pattern_config["type"] = "GaussWeightPattern"
        if previous_summand_sigmas is None:
            weight_pattern_config["sigmas"] = tf.Variable(dfpy_weight_pattern.sigmas, name=str(id(dfpy_weight_pattern)) + ".sigmas", trainable=True, constraint=lambda x: tf.math.maximum(0, x))
        else:
            weight_pattern_config["sigmas"] = tf.Variable(dfpy_weight_pattern.sigmas, name=str(id(dfpy_weight_pattern)) + ".sigmas", trainable=True, constraint=lambda x: tf.math.maximum(previous_summand_sigmas, x))

        if dfpy_weight_pattern.height >= 0:
            height = tf.Variable(dfpy_weight_pattern.height, name=str(id(dfpy_weight_pattern)) + ".height", trainable=True, constraint=lambda x: tf.math.maximum(0, x))
        if dfpy_weight_pattern.height < 0:
            height = tf.Variable(dfpy_weight_pattern.height, name=str(id(dfpy_weight_pattern)) + ".height", trainable=True, constraint=lambda x: tf.math.minimum(0, x))
        weight_pattern_config["height"] = height
        weight_pattern_config["mean"] = tf.Variable(dfpy_weight_pattern.mean)
    elif isinstance(dfpy_weight_pattern, SumWeightPattern):
        weight_pattern_config["type"] = "SumWeightPattern"
        weight_pattern_config["summands"] = []

        for i in range(len(dfpy_weight_pattern.weight_patterns)):
            summand = dfpy_weight_pattern.weight_patterns[i]
            if i > 0:
                previous_summand_sigmas = weight_pattern_config["summands"][i-1]["sigmas"]
            else:
                previous_summand_sigmas = None
            summand_weight_pattern_config = weight_pattern_config_from_dfpy_weight_pattern(summand, domain, shape, previous_summand_sigmas)
            weight_pattern_config["summands"].append(summand_weight_pattern_config)
    #elif isinstance(dfpy_weight_pattern, RepeatWeightPattern):
    #    weight_pattern_config["type"] = "RepeatWeightPattern"
    #    weight_pattern_config["weight_pattern"] = weight_pattern_config_from_dfpy_weight_pattern(dfpy_weight_pattern.weight_pattern, domain, shape)
    #    weight_pattern_config["num_repeats"] = dfpy_weight_pattern.num_repeats
    elif isinstance(dfpy_weight_pattern, RepeatedValueWeightPattern):
        weight_pattern_config["type"] = "RepeatedValueWeightPattern"
        weight_pattern_config["shape"] = dfpy_weight_pattern.shape
        weight_pattern_config["value"] = tf.Variable(float(dfpy_weight_pattern.value))
    elif isinstance(dfpy_weight_pattern, CustomWeightPattern):
        weight_pattern_config["type"] = "CustomWeightPattern"
        weight_pattern_config["pattern"] = tf.Variable(tf.convert_to_tensor(dfpy_weight_pattern.pattern))
    elif isinstance(dfpy_weight_pattern, float) or isinstance(dfpy_weight_pattern, int):
        weight_pattern_config["type"] = "ScalarWeightPattern"
        weight_pattern_config["scalar"] = tf.Variable(tf.convert_to_tensor(float(dfpy_weight_pattern)))
    elif dfpy_weight_pattern is None:
        weight_pattern_config["type"] = None
    else:
        raise RuntimeError(f"Unrecognized weight pattern type: {type(dfpy_weight_pattern)}")

    return weight_pattern_config

# TODO precomputable big tensor is pos
# gaussian tensor becomes variable
# test old version before performance-wise, using a freshly started computer


def compute_kernel_gauss_tensor(shape, domain, mean, sigmas, height):
    """Computes a tensor for the Gauss kernel.

    :param tuple of ints shape: shape of the field
    :param array_like domain: domain over which the field is defined
    :param array_like mean: mean of the Gauss
    :param array_like sigmas: sigmas of the Gauss
    :param float height: height of the Gauss
    :return Tensor gauss_tensor: a computed tensor representation of the Gauss kernel
    """
    positional_grid = compute_positional_grid(shape, domain)

    return compute_kernel_gauss_tensor_with_positional_grid(mean, sigmas, height, positional_grid)


@tf.function
def compute_kernel_gauss_tensor_with_positional_grid(mean, sigmas, height, positional_grid):
    """Computes a tensor for the Gauss kernel.

    :param array_like mean: mean of the Gauss
    :param array_like sigmas: sigmas of the Gauss
    :param float height: height of the Gauss
    :param Tensor positional_grid: positional grid
    :return Tensor gauss_tensor: a computed tensor representation of the Gauss kernel
    """
    # Define multivariate distribution over meshgrid
    mvn = tfp.distributions.MultivariateNormalDiag(
        loc=mean,
        scale_diag=sigmas)

    unnormalized = mvn.prob(positional_grid)

    normalized = unnormalized / tf.reduce_max(unnormalized)

    gauss_tensor = tf.multiply(height, normalized)

    return gauss_tensor
