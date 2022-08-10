from dff import visualization
import matplotlib.pyplot as plt


def active_nodes_table(steps, values, width, height, column_names=[], row_names=[], activation_range=(-5, 5),
                       cmap=visualization.color_maps["parula"]):

    num_columns = len(steps)
    num_rows = 1

    marker_style = dict(linestyle=':', marker='o',
                        markersize=15, markerfacecoloralt='tab:red')

    aspect_ratio = float(width)/height
    f = 3
    fig, ax = plt.subplots(figsize=(f, f/aspect_ratio), dpi=width/f)

    # Plot all fill styles.
    for row in range(num_rows):
        for column in range(num_columns):
            step = steps[column]
            value = values[steps.index(step)]
            value_normalized = (min(activation_range[1], max(activation_range[0], value)) - activation_range[0]) / (
                        activation_range[1] - activation_range[0])

            ax.plot(column, row, fillstyle='full', color=cmap(value_normalized), **marker_style)
            #ax.text(column, row, str(round(100*value)/100.0))

    #axis.set_frame_on(False)
    ax.set_xticks(range(num_columns))
    ax.set_xticklabels(column_names)
    ax.set_yticks(range(num_rows))
    ax.set_yticklabels(row_names)
    ax.set_xlim(-0.5, num_columns - 1 + 0.5)
    ax.set_ylim(num_rows - 1 + 0.5, -0.5)

    ax.tick_params(top=True, bottom=False,
                     labeltop=True, labelbottom=False)

    return fig
