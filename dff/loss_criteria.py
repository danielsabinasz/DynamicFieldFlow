import tensorflow as tf

@tf.function
def reward_positive_activation(activation, convergence_factor=5.0):
    return 1 + tf.nn.elu(convergence_factor * -activation)

@tf.function
def reward_activation_larger_than(activation, threshold, convergence_factor=5.0):
    return 1 + tf.nn.elu(convergence_factor * -(activation-threshold))

@tf.function
def reward_activation_lower_than(activation, threshold, convergence_factor=5.0):
    return 1 + tf.nn.elu(convergence_factor * (activation-threshold))


@tf.function
def reward_negative_activation(activation, convergence_factor=5.0):
    return 1 + tf.nn.elu(convergence_factor * activation)


@tf.function
def reward_zero_activation(activation, convergence_factor=5.0):
    return 1 + tf.math.abs(activation)


@tf.function
def reward_exact_activation_value(activation, value):
    return 1 + tf.math.abs(activation - value)


@tf.function
def peak_only_at_location(activation_snapshot, peak_location, radius, resting_level=-5.0):
    sum = 0.0
    field_width = activation_snapshot.shape[0]
    for i in range(field_width):
        dist = tf.math.abs(i - peak_location)
        if dist <= radius:
            sum += 1 + tf.nn.elu(5 * -(activation_snapshot[i])) / (2 * radius)
        elif dist > radius and dist < 2 * radius:
            sum += 1 + tf.nn.elu(5 * activation_snapshot[i]) / (2 * radius)

    max_activation_left = tf.reduce_max(activation_snapshot[0:peak_location - radius])
    max_activation_right = tf.reduce_max(activation_snapshot[peak_location + radius:field_width])
    max_activation_outside_peak = tf.math.maximum(max_activation_left, max_activation_right)
    sum += 1 + tf.nn.elu(5 * max_activation_outside_peak)

    return sum


@tf.function
def peak_at_location(activation_snapshot, peak_location, radius, convergence_factor=5.0):
    if peak_location >= 0:
        field_width = activation_snapshot.shape[0]

        location_outside_radius_left = max(0, peak_location - 2 * radius)
        location_outside_radius_right = min(field_width - 1, peak_location + 2 * radius)

        activation_at_peak_center = activation_snapshot[peak_location]
        activation_at_half_radius_left = activation_snapshot[peak_location - radius//2]
        activation_at_half_radius_right = activation_snapshot[peak_location + radius//2]
        activation_outside_radius_left = activation_snapshot[location_outside_radius_left]
        activation_outside_radius_right = activation_snapshot[location_outside_radius_right]

        loss1 = 8 * reward_activation_larger_than(activation_at_peak_center, threshold=0.5, convergence_factor=convergence_factor)
        loss2 = reward_positive_activation(activation_at_half_radius_left, convergence_factor=15)
        loss3 = reward_positive_activation(activation_at_half_radius_right, convergence_factor=15)
        loss4 = reward_negative_activation(activation_outside_radius_left)
        loss5 = reward_negative_activation(activation_outside_radius_right)
        #tf.print("peak loss", loss1, loss2, loss3, loss4, loss5)

        peak_loss = loss1 + loss2 + loss3 + loss4 + loss5

        peak_loss /= 12
        return peak_loss
    else:
        return 0.0


@tf.function
def no_peak_at_location(activation_snapshot, peak_location):
    if peak_location >= 0:
        return reward_negative_activation(activation_snapshot[peak_location])
    else:
        return 0.0


@tf.function
def subthreshold_bumps_at_locations(activation_snapshot, locations, distance=0.0):
    sum = 0.0
    for location in locations:
        if location >= 0:
            sum += 1 + tf.math.abs(activation_snapshot[location] + distance)
    return sum / len(locations)


@tf.function
def node_activation_at_time_step(activation_snapshots, desired_time_step, window, threshold):
    upper_bound = min(len(activation_snapshots)-1, desired_time_step + window)
    lower_bound = max(0, desired_time_step - window)

    loss1 = reward_negative_activation(activation_snapshots[lower_bound])
    loss3 = reward_activation_larger_than(activation_snapshots[upper_bound], threshold)

    loss = loss1 + loss3

    return loss

@tf.function
def peak_formation_at_location_at_time_step(activation_snapshots, desired_time_step, window, peak_location, radius):
    upper_bound = min(len(activation_snapshots)-1, desired_time_step + window)
    lower_bound = max(0, desired_time_step - window)

    loss = 0
    loss1 = no_peak_at_location(activation_snapshots[lower_bound], peak_location)
    #loss2 = reward_exact_activation_value(activation_snapshots[desired_time_step][peak_location], 0.5)
    loss3 = peak_at_location(activation_snapshots[upper_bound], peak_location, radius)

    loss = loss1 + loss3
    tf.print("loss contributions ", loss1, loss3)

    return loss


@tf.function
def match_reaction_time(activation_snapshots, desired_reaction_time, window=None):
    if window is None:
        lower_bound = 0
        upper_bound = len(activation_snapshots)
    else:
        lower_bound = desired_reaction_time - window
        upper_bound = desired_reaction_time + window
    max_activations = [tf.reduce_max(activation_snapshots[t]) for t in range(len(activation_snapshots))]
    max_activations_before_reaction_time = max_activations[lower_bound:desired_reaction_time - 1]
    max_activations_after_reaction_time = max_activations[desired_reaction_time - 1:upper_bound]
    elued_max_activations_before_reaction_time = 1 + tf.nn.elu(
        tf.convert_to_tensor(max_activations_before_reaction_time))
    elued_max_activations_after_reaction_time = 1 + tf.nn.elu(
        -tf.convert_to_tensor(max_activations_after_reaction_time))
    return tf.reduce_sum(elued_max_activations_before_reaction_time) / (
                desired_reaction_time - lower_bound) + tf.reduce_sum(elued_max_activations_after_reaction_time) / (
                upper_bound - desired_reaction_time)


@tf.function
def match_reaction_time_fast(activation_snapshots, desired_reaction_time, window=None, convergence_factor=1.0):
    if window is None:
        lower_bound = 0
        upper_bound = len(activation_snapshots)
    else:
        lower_bound = desired_reaction_time - window
        upper_bound = desired_reaction_time + window
    max_activations = [tf.reduce_max(activation_snapshots[t]) for t in range(len(activation_snapshots))]
    max_activation_at_lower_bound = max_activations[lower_bound]
    max_activation_at_rt = max_activations[desired_reaction_time]
    max_activation_at_upper_bound = max_activations[upper_bound]
    loss_lb = reward_negative_activation(max_activation_at_lower_bound, convergence_factor)
    loss_rt = reward_zero_activation(max_activation_at_rt, convergence_factor)
    loss_ub = reward_positive_activation(max_activation_at_upper_bound, convergence_factor)
    #tf.print("loss_lb", loss_lb)
    #tf.print("loss_rt", loss_rt)
    #tf.print("loss_ub", loss_ub)
    #tf.print("max_activation_at_lower_bound", max_activation_at_lower_bound)
    #tf.print("max_activation_at_rt", max_activation_at_rt)
    #tf.print("max_activation_at_upper_bound", max_activation_at_upper_bound)
    return 0.33 * loss_lb + 0.33 * loss_rt + 0.33 * loss_ub
