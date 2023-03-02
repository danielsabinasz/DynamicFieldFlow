import sys
#print("RUNNING ", sys.argv[1])
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from dfpy import *
from dff import *
import tensorflow as tf
import matplotlib.pyplot as plt

t_max=int(600)
time_step_duration=int(20)
num_time_steps=int(t_max/time_step_duration)
num_epochs = 100

gauss1 = GaussInput(
    dimensions=[Dimension(0, 50, 51)],
    mean=[10.0],
    height=6.0,
    sigmas=[3.0]
)

gauss2 = GaussInput(
    dimensions=[Dimension(0, 50, 51)],
    mean=[40.0],
    height=6.0,
    sigmas=[3.0]
)

field = Field(
    dimensions=[Dimension(0, 50, 51)],
    resting_level=-5.0,
    interaction_kernel=SumWeightPattern([
        GaussWeightPattern(height=1.0, sigmas=(3.0,)),
        GaussWeightPattern(height=-0.1, sigmas=(5.0,))
    ]),
    global_inhibition=0.0
)

connect(gauss1, field)
connect(gauss2, field)


dff_simulator = Simulator(time_step_duration=time_step_duration)
ns= dfpy.shared.get_default_neural_structure()


simulation_call = dff_simulator.get_unrolled_simulation_call_with_history(num_time_steps)

@tf.function
def peak_at_location(activation_snapshot, peak_location, radius, resting_level=-5.0):
    sum = 0.0
    for i in range(activation_snapshot.shape[0]):
        dist = tf.math.abs(i-peak_location)
        if dist <= radius:
            sum += 1 + tf.nn.elu(5*-(activation_snapshot[i]))
        elif dist > radius and dist < 2*radius:
            sum += 1 + tf.nn.elu(5*activation_snapshot[i])
    return sum / (2*radius)


@tf.function
def subthreshold_bumps_at_locations(activation_snapshot, locations, distance=0.0):
    sum = 0.0
    for location in locations:
        sum += 1 + tf.nn.elu(tf.math.abs(activation_snapshot[location]+distance))
    return sum/len(locations)


@tf.function
def smooth_minimum(x):
    sum_nominator = 0.0
    sum_denominator = 0.0
    for i in range(len(x)):
        sum_nominator += x[i] * tf.math.exp(-x[i])
        sum_denominator += tf.math.exp(-x[i])
    return sum_nominator / sum_denominator


@tf.function
def smooth_maximum(x):
    sum_nominator = 0.0
    sum_denominator = 0.0
    for i in range(len(x)):
        sum_nominator += x[i] * tf.math.exp(x[i])
        sum_denominator += tf.math.exp(x[i])
    return sum_nominator / sum_denominator


@tf.function
def smooth_argmax(x):
    normalizer = tf.reduce_sum(tf.math.exp(x))
    sum = 0.0
    for i in range(len(x)):
        sum += tf.math.exp(x[i]) / normalizer * i
    return sum


@tf.function
def peak_only_at_location(activation_snapshot, peak_location, radius, resting_level=-5.0):
    sum = 0.0
    field_width = activation_snapshot.shape[0]
    for i in range(field_width):
        dist = tf.math.abs(i-peak_location)
        if dist <= radius:
            sum += (1 + tf.nn.elu(10*-(activation_snapshot[i])))/(2*radius)
        elif dist > radius:
            sum += (1 + tf.nn.elu(10*activation_snapshot[i]))/(field_width-2*radius)
    return sum


@tf.function
def selective_peak_at_any_one_location(activation_snapshot, peak_locations, radius, resting_level=-5.0):
    # Compute candidate losses for a peak at any one location
    candidate_losses = []
    for peak_location in peak_locations:
        candidate_loss = peak_only_at_location(activation_snapshot, peak_location, radius, resting_level)
        candidate_losses.append(candidate_loss)

    loss = tf.reduce_min(candidate_losses)

    return loss

@tf.function
def match_reaction_time(activation_snapshots, desired_reaction_time, window=None):
    if window is None:
        lower_bound = 0
        upper_bound = len(activation_snapshots)
    else:
        lower_bound = desired_reaction_time-window
        upper_bound = desired_reaction_time+window
    max_activations = [tf.reduce_max(activation_snapshots[t]) for t in range(len(activation_snapshots))]
    max_activations_before_reaction_time = max_activations[lower_bound:desired_reaction_time-1]
    max_activations_after_reaction_time = max_activations[desired_reaction_time-1:upper_bound]
    elued_max_activations_before_reaction_time = 1+tf.nn.elu(tf.convert_to_tensor(max_activations_before_reaction_time))
    elued_max_activations_after_reaction_time = 1+tf.nn.elu(-tf.convert_to_tensor(max_activations_after_reaction_time))
    return tf.reduce_sum(elued_max_activations_before_reaction_time) / (desired_reaction_time-lower_bound) + tf.reduce_sum(elued_max_activations_after_reaction_time) / (upper_bound - desired_reaction_time)


@tf.function
def reward_positive_activation(activation, convergence_factor=5.0):
    return 1+tf.nn.elu(convergence_factor*-activation)

@tf.function
def reward_negative_activation(activation, convergence_factor=5.0):
    return 1+tf.nn.elu(convergence_factor*activation)

@tf.function
def reward_zero_activation(activation, convergence_factor=5.0):
    return 1 + tf.nn.elu(convergence_factor*tf.math.abs(activation))


@tf.function
def peak_at_location_fast(activation_snapshot, peak_location, radius):
    field_width = activation_snapshot.shape[0]

    location_outside_radius_left = max(0, peak_location - 2 * radius)
    location_outside_radius_right = min(field_width - 1, peak_location + 2 * radius)

    activation_at_peak_center = activation_snapshot[peak_location]
    activation_at_radius_left = activation_snapshot[peak_location - radius]
    activation_at_radius_right = activation_snapshot[peak_location + radius]
    activation_outside_radius_left = activation_snapshot[location_outside_radius_left]
    activation_outside_radius_right = activation_snapshot[location_outside_radius_right]

    peak_loss = 8*reward_positive_activation(activation_at_peak_center) + reward_zero_activation(
        activation_at_radius_left) + reward_zero_activation(
        activation_at_radius_right) + reward_negative_activation(
        activation_outside_radius_left) + reward_negative_activation(activation_outside_radius_right)

    peak_loss /= 12
    return peak_loss

@tf.function
def no_peak_at_location(activation_snapshot, peak_location):
    return reward_negative_activation(activation_snapshot[peak_location])

@tf.function
def loss():
    time_invariant_variable_variant_tensors = dff_simulator.compute_time_invariant_variable_variant_tensors()
    initial_values = dff_simulator.update_initial_values(time_invariant_variable_variant_tensors)

    # Perform simulation
    total_loss = 0

    for i in range(100):
        values_history = simulation_call(0, initial_values, time_invariant_variable_variant_tensors)
        activation_snapshot = values_history[-1][2]
        if i >= 50:
            total_loss += peak_at_location_fast(activation_snapshot, peak_location=10, radius=3)
            total_loss += no_peak_at_location(activation_snapshot, peak_location=40)
        else:
            total_loss += peak_at_location_fast(activation_snapshot, peak_location=40, radius=3)
            total_loss += no_peak_at_location(activation_snapshot, peak_location=10)

    """for i in range(70):
        values_history = simulation_call(0, initial_values, time_invariant_variable_variant_tensors)
        total_loss += peak_only_at_location(
            activation_snapshot=values_history[-1][2],
            peak_location=10.0,
            radius=5
        )

    for i in range(30):
        values_history = simulation_call(0, initial_values, time_invariant_variable_variant_tensors)
        total_loss += peak_only_at_location(
            activation_snapshot=values_history[-1][2],
            peak_location=40.0,
            radius=5
        )"""

    #total_loss += peak_at_location(
    #    activation_snapshot=values_history[-1][2],
    #    peak_location=40.0,
    #    radius=5
    #)


    return total_loss


# Create an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.2)


losses = []
resting_levels = []
global_inhibitions = []
local_exc_heights = []
local_exc_sigmas = []
mid_range_inhibition_heights = []
mid_range_inhibition_sigmas = []
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        ls = loss()
    losses.append(ls)

    print(f"training epoch {epoch}")
    print(f"loss {ls}")

    gauss1_height = dff_simulator.variables[gauss1]["height"]
    gauss1_sigmas = dff_simulator.variables[gauss1]["sigmas"]
    gauss2_height = dff_simulator.variables[gauss2]["height"]
    gauss2_sigmas = dff_simulator.variables[gauss2]["sigmas"]

    resting_level = dff_simulator.variables[field]["resting_level"]
    global_inhibition = dff_simulator.variables[field]["global_inhibition"]
    local_exc_height = dff_simulator.variables[field]["interaction_kernel_weight_pattern_config"]["summands"][0]["height"]
    local_exc_sigma = dff_simulator.variables[field]["interaction_kernel_weight_pattern_config"]["summands"][0]["sigmas"]
    mid_range_inh_height = dff_simulator.variables[field]["interaction_kernel_weight_pattern_config"]["summands"][1]["height"]
    mid_range_inh_sigma = dff_simulator.variables[field]["interaction_kernel_weight_pattern_config"]["summands"][1]["sigmas"]
    print(f"resting_level={resting_level.numpy()}")
    print(f"global_inhibition={global_inhibition.numpy()}")
    print(f"local_exc_height={local_exc_height.numpy()}")
    print(f"local_exc_sigma={local_exc_sigma.numpy()}")
    print(f"mid_range_inh_height={mid_range_inh_height.numpy()}")
    print(f"mid_range_inh_sigma={mid_range_inh_sigma.numpy()}")
    print(f"gauss1_height={gauss1_height.numpy()}")
    print(f"gauss1_sigmas={gauss1_sigmas.numpy()}")
    print(f"gauss2_height={gauss2_height.numpy()}")
    print(f"gauss2_sigmas={gauss2_sigmas.numpy()}")

    resting_levels.append(resting_level.numpy())
    global_inhibitions.append(global_inhibition.numpy())
    local_exc_heights.append(local_exc_height.numpy())
    local_exc_sigmas.append(local_exc_sigma.numpy())
    mid_range_inhibition_heights.append(mid_range_inh_height.numpy())
    mid_range_inhibition_sigmas.append(mid_range_inh_sigma.numpy())

    trainable_vars = [resting_level, global_inhibition, local_exc_height, local_exc_sigma, mid_range_inh_height, mid_range_inh_sigma, gauss1_height, gauss1_sigmas, gauss2_height, gauss2_sigmas]
    gradients = tape.gradient(ls, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))


time_invariant_variable_variant_tensors = dff_simulator.compute_time_invariant_variable_variant_tensors()
initial_values = dff_simulator.update_initial_values(time_invariant_variable_variant_tensors)
values_history = simulation_call(0, initial_values, time_invariant_variable_variant_tensors)

skip = 1
figure, axis = plt.subplots(1, num_time_steps//skip+1)
figure.subplots_adjust(top=0.8)
figure.set_figwidth(2 * num_time_steps//skip+1)
figure.set_figheight(2)
for t in range(num_time_steps+1):
    if t % skip != 0:
        continue
    axis[t//skip].grid()
    axis[t//skip].set_title(f"t{t}")
    axis[t//skip].set_xticks([0, 10, 20, 30, 40, 50])
    axis[t//skip].set_xticklabels(["0", "10", "20", "30", "40", "50"])
    axis[t//skip].set_ylim(-15, 15)
    axis[t//skip].plot(values_history[t][2])

plt.savefig("time_course.png")
plt.show()


left_count = 0
right_count = 0
both_count = 0
none_count = 0
for n in range(100):
    values_history = simulation_call(0, initial_values, time_invariant_variable_variant_tensors)
    activation_left = tf.reduce_max(values_history[-1][2][0:25]).numpy()
    activation_right = tf.reduce_max(values_history[-1][2][26:51]).numpy()
    if activation_left > 0 and activation_right < 0:
        left_count += 1
    elif activation_right > 0 and activation_left < 0:
        right_count += 1
    elif activation_left > 0 and activation_right > 0:
        both_count += 1
    elif activation_left < 0 and activation_right < 0:
        none_count += 1

print("left", left_count, "right", right_count, "both", both_count, "none", none_count)

skip = 1
figure, axis = plt.subplots(1, num_time_steps//skip+1)
figure.subplots_adjust(top=0.8)
figure.set_figwidth(2 * num_time_steps//skip+1)
figure.set_figheight(2)
for t in range(num_time_steps+1):
    if t % skip != 0:
        continue
    axis[t//skip].grid()
    axis[t//skip].set_title(f"t{t}")
    axis[t//skip].set_xticks([0, 10, 20, 30, 40, 50])
    axis[t//skip].set_xticklabels(["0", "10", "20", "30", "40", "50"])
    axis[t//skip].set_ylim(-15, 15)
    axis[t//skip].plot(values_history[t][2])

plt.savefig("time_course.png")
plt.show()


#
# Plot loss and parameters over epoch
#

plt.figure(figsize=(10,2))
plt.title("Loss")
plt.xlabel("Epoch")
plt.plot(losses)
plt.savefig("loss.png")
plt.show()

plt.figure(figsize=(10,2))
plt.title("Resting level")
plt.xlabel("Epoch")
plt.plot(resting_levels)
plt.savefig("resting_level.png")
plt.show()

plt.figure(figsize=(10,2))
plt.title("Global inhibition")
plt.xlabel("Epoch")
plt.plot(global_inhibitions)
plt.savefig("global_inhibition.png")
plt.show()

plt.figure(figsize=(10,2))
plt.title("Local excitation height")
plt.xlabel("Epoch")
plt.plot(local_exc_heights)
plt.savefig("local_exc_height.png")
plt.show()

plt.figure(figsize=(10,2))
plt.title("Local excitation sigma")
plt.xlabel("Epoch")
plt.plot(local_exc_sigmas)
plt.savefig("local_exc_sigma.png")
plt.show()

plt.figure(figsize=(10,2))
plt.title("Mid-range inhibition height")
plt.xlabel("Epoch")
plt.plot(mid_range_inhibition_heights)
plt.savefig("mid_range_inh_height.png")
plt.show()

plt.figure(figsize=(10,2))
plt.title("Mid-range inhibition sigma")
plt.xlabel("Epoch")
plt.plot(mid_range_inhibition_sigmas)
plt.savefig("mid_range_inh_sigma.png")
plt.show()

