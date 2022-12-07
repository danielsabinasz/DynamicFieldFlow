from dfpy import *
from dff import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

t_max=int(600)
time_step_duration=int(20)
num_time_steps=int(t_max/time_step_duration)
num_epochs = 200

# Prepare custom input time course
X = np.arange(0, 51, 1)
def gaussian(x, mean):
    return np.exp( -np.power((x-mean)/2, 2) )
input_blank = [ [0.0] * 51 ]
input_cue = 4*gaussian(X, 10) + 4*gaussian(X, 40)
input_target = 6*gaussian(X, 10)

timed_custom_input = TimedCustomInput(
    dimensions=[Dimension(0, 50, 51)],
    timed_custom_input=
        np.concatenate([
            np.tile(input_blank, (5,1)),
            np.tile(input_cue, (5,1)),
            np.tile(input_blank, (5,1)),
            np.tile(input_target, (15,1))
        ])
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
            sum += tf.nn.elu(-(activation_snapshot[i]))
        elif dist > radius and dist < 2*radius:
            sum += tf.nn.elu(activation_snapshot[i])
    return sum


@tf.function
def subthreshold_bumps_at_locations(activation_snapshot, locations, distance=0.0):
    sum = 0.0
    for location in locations:
        sum += tf.nn.elu(tf.math.square(activation_snapshot[location]+distance))
    return sum


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
def selective_peak_at_locations(activation_snapshot, peak_locations, radius, resting_level=-5.0):
    # Compute candidate losses for a peak at any one location
    candidate_losses = []
    for peak_location in peak_locations:
        candidate_loss = peak_at_location(activation_snapshot, peak_location, radius, resting_level)
        candidate_losses.append(candidate_loss)

    loss = smooth_minimum(candidate_losses)

    return loss

@tf.function
def match_reaction_time(activation_snapshots, desired_reaction_time, window=None):
    if window is None:
        lower_bound = 0
        upper_bound = len(activation_snapshots)
    else:
        lower_bound = desired_reaction_time-window
        upper_bound = desired_reaction_time+window
    normalized_activations = [tf.math.tanh(tf.reduce_max(activation_snapshots[t])) for t in range(len(activation_snapshots))]
    nma_before_reaction_time = normalized_activations[lower_bound:desired_reaction_time-1]
    nma_after_reaction_time = normalized_activations[desired_reaction_time-1:upper_bound]
    return tf.reduce_sum(nma_before_reaction_time) / (desired_reaction_time-lower_bound) - tf.reduce_sum(nma_after_reaction_time) / (upper_bound - desired_reaction_time)

@tf.function
def loss():
    time_invariant_variable_variant_tensors = dff_simulator.compute_time_invariant_variable_variant_tensors()
    initial_values = dff_simulator.update_initial_values(time_invariant_variable_variant_tensors)

    # Perform simulation
    values_history = simulation_call(0, initial_values, time_invariant_variable_variant_tensors)

    loss_bumps = subthreshold_bumps_at_locations(
        activation_snapshot=values_history[10][1],
        locations=[10, 40],
        distance=1.0
    )
    loss_peak = peak_at_location(
        activation_snapshot=values_history[-1][1],
        peak_location=10,
        radius=5
    )
    loss_rt = match_reaction_time(
        activation_snapshots=[values_history[t][1] for t in range(len(values_history))],
        desired_reaction_time=25,
        window=5
    )

    total_loss = 2*loss_bumps + 0.5*loss_peak + 30*loss_rt
    tf.print(loss_bumps, loss_peak, loss_rt, loss_peak/loss_rt)

    return total_loss


# Create an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.2)



final_activation_figure, final_activation_axis = plt.subplots(1, num_epochs)
final_activation_figure.subplots_adjust(top=0.8)
final_activation_figure.set_figwidth(num_epochs)
final_activation_figure.set_figheight(1)

losses = []
resting_levels = []
global_inhibitions = []
local_excitations = []
mid_range_inhibitions = []
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        ls = loss()
    losses.append(ls)

    print(f"training epoch {epoch}")
    print(f"loss {ls}")

    resting_level = dff_simulator.variables[field]["resting_level"]
    global_inhibition = dff_simulator.variables[field]["global_inhibition"]
    local_exc = dff_simulator.variables[field]["interaction_kernel_weight_pattern_config"]["summands"][0]["height"]
    mid_range_inh = dff_simulator.variables[field]["interaction_kernel_weight_pattern_config"]["summands"][1]["height"]
    print(f"resting_level={resting_level.numpy()}")
    print(f"global_inhibition={global_inhibition.numpy()}")
    print(f"local_exc={local_exc.numpy()}")
    print(f"mid_range_inh={mid_range_inh.numpy()}")

    resting_levels.append(resting_level.numpy())
    global_inhibitions.append(global_inhibition.numpy())
    local_excitations.append(local_exc.numpy())
    mid_range_inhibitions.append(mid_range_inh.numpy())

    trainable_vars = [resting_level, global_inhibition, local_exc, mid_range_inh]
    gradients = tape.gradient(ls, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))

    if epoch % 1 == 0:
        col = epoch//1
        time_invariant_variable_variant_tensors = dff_simulator.compute_time_invariant_variable_variant_tensors()
        values_history = simulation_call(0, dff_simulator.initial_values, time_invariant_variable_variant_tensors)
        final_activation_axis[col].set_title(f"ep. {epoch}")
        final_activation_axis[col].set_ylim(-5, 5)
        final_activation_axis[col].plot(values_history[-1][1])


time_invariant_variable_variant_tensors = dff_simulator.compute_time_invariant_variable_variant_tensors()
initial_values = dff_simulator.update_initial_values(time_invariant_variable_variant_tensors)
values_history = simulation_call(0, initial_values, time_invariant_variable_variant_tensors)

# How many time snapshots should be shown? e.g., skip=5: every 5th snapshot, skip=1: every snapshot
skip = 1

# Plot time snapshots
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
    axis[t//skip].plot(values_history[t][1])
    axis[t//skip].plot(values_history[t][0])
plt.show()


#
# Plot loss and parameters over epoch
#

plt.figure(figsize=(10,2))
plt.title("Loss")
plt.xlabel("Epoch")
plt.plot(losses)
plt.show()

plt.figure(figsize=(10,2))
plt.title("Resting level")
plt.xlabel("Epoch")
plt.plot(resting_levels)
plt.show()

plt.figure(figsize=(10,2))
plt.title("Global inhibition")
plt.xlabel("Epoch")
plt.plot(global_inhibitions)
plt.show()

plt.figure(figsize=(10,2))
plt.title("Local excitation")
plt.xlabel("Epoch")
plt.plot(local_excitations)
plt.show()

plt.figure(figsize=(10,2))
plt.title("Mid-range inhibition")
plt.xlabel("Epoch")
plt.plot(mid_range_inhibitions)
plt.show()

