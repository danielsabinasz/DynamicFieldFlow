def active_nodes_3d_grid(axis, column_names, row_names, depth_modifiers, steps, simulator, time_step):
    num_columns = len(column_names)
    num_rows = len(row_names)

    # Plot all fill styles.
    for row in range(num_rows):
        for column in range(num_columns):
            num_depth = len(depth_modifiers[row])
            for depth in range(num_depth - 1, -1, -1):
                depth_modifier = depth_modifiers[row][depth]
                if "color" not in depth_modifier:
                    depth_modifier["color"] = "blue"
                if "marker" not in depth_modifier:
                    depth_modifier["marker"] = "o"

                marker_style = dict(color='tab:' + depth_modifier["color"], linestyle=':',
                                    marker=depth_modifier["marker"], markersize=15,
                                    markerfacecoloralt='tab:' + depth_modifier["color"])
                value = simulator.get_recorded_value_at_time_step(steps[row][column][depth], time_step)

                offset = 0.5 * (float(depth) / (num_depth - 1) - 0.5)

                if value > 0:
                    axis.plot(column + offset, row - offset, fillstyle='full', **marker_style)
                else:
                    axis.plot(column + offset, row - offset, fillstyle='none', **marker_style)

    axis.set_frame_on(False)
    axis.set_xticks(range(num_columns))
    axis.set_yticks(range(num_rows))
    axis.set_yticklabels(row_names)
    axis.set_xlim(-0.5, num_columns - 1 + 0.5)
    axis.set_ylim(num_rows - 1 + 0.5, -0.5)

    axis.tick_params(top=True, bottom=False,
                     labeltop=True, labelbottom=False)
