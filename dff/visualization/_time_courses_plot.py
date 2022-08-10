import matplotlib.pyplot as plt


def time_courses_plot(steps, simulator):
    domain = steps[0].domain
    shape = steps[0].shape
    ndim = len(domain)

    if ndim == 0:
        fig, ax = plt.subplots()
        for i in range(len(steps)):
            step = steps[i]
            time_course = simulator.get_recorded_values_for_step(step)
            im = ax.plot(range(0, len(time_course)), time_course, label=str(i))
        ax.legend()
        return fig
