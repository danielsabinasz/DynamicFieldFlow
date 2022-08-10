import matplotlib.pyplot as plt


def image_plot(step, simulator):
    """Creates a snapshot plot of an image.

    :param Step step: the step to visualize
    :param Simulator simulator: the simulator that was used to compute the activation
    """

    domain = step.domain

    value = simulator.get_value(step)

    fig, ax = plt.subplots()

    ax.imshow(value,
                   extent=[domain[0][0], domain[0][1], domain[1][0], domain[1][1]])

    return fig
