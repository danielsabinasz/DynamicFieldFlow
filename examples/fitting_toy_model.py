from dfpy import *
from dff import *
import tensorflow as tf
import matplotlib.pyplot as plt

t_max=int(600)
time_step_duration=int(20)
num_time_steps=int(t_max/time_step_duration)
num_epochs = 50

gauss1 = GaussInput(
    dimensions=[Dimension(0, 50, 51)],
    mean=[10.0],
    height=5.0,
    sigmas=[3.0]
)

gauss2 = GaussInput(
    dimensions=[Dimension(0, 50, 51)],
    mean=[40.0],
    height=0.0,
    sigmas=[3.0]
)

field = Field(
    dimensions=[Dimension(0, 50, 51)],
    resting_level=-5.0,
    interaction_kernel=SumWeightPattern([
        GaussWeightPattern(height=1.0, sigmas=(3.0, 3.0)),
        GaussWeightPattern(height=-0.1, sigmas=(5.0, 5.0))
    ]),
    global_inhibition=0.0
)

connect(gauss1, field)
connect(gauss2, field)


dff_simulator = Simulator(time_step_duration=time_step_duration)
ns= dfpy.shared.get_default_neural_structure()


simulation_call = dff_simulator.get_unrolled_simulation_call_with_history(num_time_steps)


@tf.function
def peak_at_location(activation_snapshot, peak_location, d_max, resting_level=-5.0):
    sum = 0.0
    abs_resting_level = abs(resting_level)
    for i in range(activation_snapshot.shape[0]):
        dist = tf.math.abs(i-peak_location)
        if dist <= d_max:
            sum += tf.nn.elu(-(activation_snapshot[i]))
        elif dist > d_max and dist < 2*d_max:
            sum += tf.nn.elu(activation_snapshot[i])
    return sum


@tf.function
def subthreshold_bumps_at_locations(activation_snapshot, locations):
    sum = 0.0
    for location in locations:
        sum += tf.nn.elu(tf.math.square(activation_snapshot[location]))
        #for i in range(activation_snapshot.shape[0]):
        #    dist = tf.math.abs(i-location)
        #    if dist == 0:
        #        sum += tf.nn.elu(tf.math.square(activation_snapshot[i]))
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
    #slope = (normalizeradius_activations[upper_bound]-normalizeradius_activations[lower_bound])/(upper_bound-lower_bound)
    #return -slope
    return tf.reduce_sum(nma_before_reaction_time) / (desired_reaction_time-lower_bound) - tf.reduce_sum(nma_after_reaction_time) / (upper_bound - desired_reaction_time)

@tf.function
def loss():
    time_invariant_variable_variant_tensors = dff_simulator.compute_time_invariant_variable_variant_tensors()
    initial_values = dff_simulator.update_initial_values(time_invariant_variable_variant_tensors)

    # Perform simulation
    values_history = simulation_call(0, initial_values, time_invariant_variable_variant_tensors)

    #total_loss = subthreshold_bumps_at_locations(values_history[30][2], [10, 40])
    #total_loss = peak_at_location(values_history[30][2], 10, 5)

    total_loss = match_reaction_time([values_history[t][2] for t in range(len(values_history))], 20, 10)

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
    print(f"resting_level={resting_level}")
    print(f"global_inhibition={global_inhibition}")
    print(f"local_exc={local_exc}")
    print(f"mid_range_inh={mid_range_inh}")

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
        final_activation_axis[col].plot(values_history[-1][2])

    #    gauss_distribution = dff_simulator.initial_values[0] + dff_simulator.initial_values[1]
    #    gauss_distribution_shifted = gauss_distribution - tf.ones(gauss_distribution.shape)*3
    #    final_activation_axis[col].plot(gauss_distribution_shifted)


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

plt.show()



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

