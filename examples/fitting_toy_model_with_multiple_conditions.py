from dfpy import *
from dff import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

t_max=int(600)
time_step_duration=int(20)
num_time_steps=int(t_max/time_step_duration)
num_epochs = 200

#
# Prepare custom input time course
#

def gaussian(x, mean):
    return np.exp( -np.power((x-mean)/2, 2) )

X = np.arange(0, 51, 1)

input_blank = [ [0.0] * 51 ]

custom_inputs_by_condition = {
    "valid":
        np.concatenate([
            np.tile(input_blank, (5,1)), # fixation
            np.tile(4*gaussian(X, 10), (5,1)), # cue
            np.tile(input_blank, (5,1)), # blank
            np.tile(6*gaussian(X, 10), (15,1)) # target
        ]),
    "invalid":
        np.concatenate([
            np.tile(input_blank, (5,1)), # fixation
            np.tile(4*gaussian(X, 40), (5,1)), # cue
            np.tile(input_blank, (5,1)), # blank
            np.tile(6*gaussian(X, 10), (15,1)) # target
        ]),
    "double":
        np.concatenate([
            np.tile(input_blank, (5,1)), # fixation
            np.tile(4*gaussian(X, 10) + 4*gaussian(X, 40), (5,1)), # cue
            np.tile(input_blank, (5,1)), # blank
            np.tile(6*gaussian(X, 40), (15,1)) # target
        ])
}
expected_subthreshold_bumps_by_condition = {
    "valid": [10],
    "invalid": [40],
    "double": [10, 40]
}
expected_target_peak_by_condition = {
    "valid": 10,
    "invalid": 10,
    "double": 40
}

timed_custom_input = TimedCustomInput(
    dimensions=[Dimension(0, 50, 51)],
    timed_custom_input=custom_inputs_by_condition["double"]
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

connect(timed_custom_input, field)


dff_simulator = Simulator(time_step_duration=time_step_duration)
ns = dfpy.shared.get_default_neural_structure()


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
def smooth_argmax(x):
    normalizer = tf.reduce_sum(tf.math.exp(x))
    sum = 0.0
    for i in range(len(x)):
        sum += tf.math.exp(x[i]) / normalizer * i
    return sum


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
def loss():
    total_loss = 0
    for condition in custom_inputs_by_condition:
        dff_simulator.variables[timed_custom_input]["timed_custom_input"].assign(custom_inputs_by_condition[condition])

        time_invariant_variable_variant_tensors = dff_simulator.compute_time_invariant_variable_variant_tensors()
        initial_values = dff_simulator.update_initial_values(time_invariant_variable_variant_tensors)

        # Perform simulation
        values_history = simulation_call(0, initial_values, time_invariant_variable_variant_tensors)

        loss_bumps = subthreshold_bumps_at_locations(
            activation_snapshot=values_history[10][1],
            locations=expected_subthreshold_bumps_by_condition[condition],
            distance=1.0
        )
        loss_peak = peak_at_location(
            activation_snapshot=values_history[-1][1],
            peak_location=expected_target_peak_by_condition[condition],
            radius=5
        )
        loss_rt = match_reaction_time(
            activation_snapshots=[values_history[t][1] for t in range(len(values_history))],
            desired_reaction_time=20,
            window=5
        )

        loss_for_condition = loss_bumps + loss_peak + loss_rt
        tf.print("loss for condition", condition, loss_for_condition)
        total_loss += loss_for_condition

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

    resting_levels.append(resting_level.numpy())
    global_inhibitions.append(global_inhibition.numpy())
    local_exc_heights.append(local_exc_height.numpy())
    local_exc_sigmas.append(local_exc_sigma.numpy())
    mid_range_inhibition_heights.append(mid_range_inh_height.numpy())
    mid_range_inhibition_sigmas.append(mid_range_inh_sigma.numpy())

    trainable_vars = [resting_level, global_inhibition, local_exc_height, local_exc_sigma, mid_range_inh_height, mid_range_inh_sigma]
    gradients = tape.gradient(ls, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))

#
# Plot time snapshots for every condition
#

# How many time snapshots should be shown? e.g., skip=5: every 5th snapshot, skip=1: every snapshot
skip = 1

for condition in custom_inputs_by_condition:
    dff_simulator.variables[timed_custom_input]["timed_custom_input"].assign(custom_inputs_by_condition[condition])
    time_invariant_variable_variant_tensors = dff_simulator.compute_time_invariant_variable_variant_tensors()
    initial_values = dff_simulator.update_initial_values(time_invariant_variable_variant_tensors)
    values_history = simulation_call(0, initial_values, time_invariant_variable_variant_tensors)

    figure, axis = plt.subplots(1, num_time_steps//skip+1)
    figure.subplots_adjust(top=0.8)
    figure.set_figwidth(2 * num_time_steps//skip+1)
    figure.set_figheight(2)
    figure.suptitle("Time course " + condition)
    for t in range(num_time_steps+1):
        if t % skip != 0:
            continue
        axis[t//skip].grid()
        axis[t//skip].set_title(f"t{t}")
        axis[t//skip].set_xticks([0, 10, 20, 30, 40, 50])
        axis[t//skip].set_xticklabels(["0", "10", "20", "30", "40", "50"])
        axis[t//skip].set_ylim(-15, 15)
        axis[t//skip].plot(values_history[t][1])
        axis[t//skip].plot(values_history[t][0])
    plt.savefig("time_course_" + condition + ".png")

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

