import matplotlib.pyplot as plt


def time_course_plot(step, simulator):
    time_course = simulator.get_recorded_values_for_step(step)

    fig, ax = plt.subplots()
    im = ax.plot(range(0, len(time_course)), time_course)
    return fig
