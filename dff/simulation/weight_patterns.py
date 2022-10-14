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


def weight_pattern_config_from_dfpy_weight_pattern(dfpy_weight_pattern, domain, shape):
    #domain = tf.convert_to_tensor([[dimension.lower, dimension.upper] for dimension in dfpy_field.dimensions],
    #                              dtype=tf.float32)
    #shape = tf.convert_to_tensor([dimension.size for dimension in dfpy_field.dimensions], dtype=tf.int32)

    weight_pattern_config = {
        "shape": shape,
        "domain": domain
    }

    if isinstance(dfpy_weight_pattern, GaussWeightPattern):
        weight_pattern_config["type"] = "GaussWeightPattern"
        weight_pattern_config["sigmas"] = tf.constant(dfpy_weight_pattern.sigmas)
        weight_pattern_config["height"] = tf.Variable(dfpy_weight_pattern.height, name="GaussWeightPattern.height", trainable=True)
        weight_pattern_config["mean"] = tf.constant(dfpy_weight_pattern.mean)
    elif isinstance(dfpy_weight_pattern, SumWeightPattern):
        weight_pattern_config["type"] = "SumWeightPattern"
        weight_pattern_config["summands"] = []
        for summand in dfpy_weight_pattern.weight_patterns:
            weight_pattern_config["summands"].append(
                weight_pattern_config_from_dfpy_weight_pattern(summand, domain, shape)
            )
    elif isinstance(dfpy_weight_pattern, RepeatedValueWeightPattern):
        weight_pattern_config["type"] = "RepeatedValueWeightPattern"
        weight_pattern_config["shape"] = dfpy_weight_pattern.shape
        weight_pattern_config["value"] = float(dfpy_weight_pattern.value)
    elif isinstance(dfpy_weight_pattern, CustomWeightPattern):
        weight_pattern_config["type"] = "CustomWeightPattern"
        weight_pattern_config["pattern"] = tf.convert_to_tensor(dfpy_weight_pattern.pattern)
    elif isinstance(dfpy_weight_pattern, float) or isinstance(dfpy_weight_pattern, int):
        weight_pattern_config["type"] = "ScalarWeightPattern"
        weight_pattern_config["scalar"] = tf.convert_to_tensor(float(dfpy_weight_pattern))
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
        loc=tf.reverse(mean, [0]),
        scale_diag=sigmas)

    unnormalized = mvn.prob(positional_grid)

    normalized = unnormalized / tf.reduce_max(unnormalized)

    gauss_tensor = tf.multiply(height, normalized)

    print(height, normalized)

    return gauss_tensor
