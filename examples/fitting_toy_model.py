from dfpy import *
from dff import *
import tensorflow as tf
import matplotlib.pyplot as plt

t_max=int(600)
time_step_duration=int(20)
num_time_steps=int(t_max/time_step_duration)

gauss1 = GaussInput(
    dimensions=[Dimension(0, 50, 51)],
    mean=[15.0],
    height=3.0,
    sigmas=[3.0]
)

gauss2 = GaussInput(
    dimensions=[Dimension(0, 50, 51)],
    mean=[35.0],
    height=3.0,
    sigmas=[3.0]
)

field = Field(
    dimensions=[Dimension(0, 50, 51)],
    resting_level=-5.0,
    interaction_kernel=SumWeightPattern([
        GaussWeightPattern(height=1.0, sigmas=(3.0, 3.0)),
        GaussWeightPattern(height=0.0, sigmas=(5.0, 5.0))
    ]),
    global_inhibition=0.0
)

connect(gauss1, field)
connect(gauss2, field)


dff_simulator = Simulator(time_step_duration=time_step_duration)
ns= dfpy.shared.get_default_neural_structure()


simulation_call = dff_simulator.get_unrolled_simulation_call_with_history(num_time_steps)

@tf.function
def loss():
    time_invariant_variable_variant_tensors = dff_simulator.compute_time_invariant_variable_variant_tensors()
    initial_values = dff_simulator.update_initial_values(time_invariant_variable_variant_tensors)

    # Perform simulation
    values_history = simulation_call(0, initial_values, time_invariant_variable_variant_tensors)

    gauss_distribution = initial_values[0] + initial_values[1]
    gauss_distribution_shifted = gauss_distribution - tf.ones(gauss_distribution.shape)*2.5

    total_loss = tf.math.square(tf.reduce_sum(values_history[-1][2] - gauss_distribution_shifted))
    return total_loss


# Create an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.2)


num_epochs = 100

figure, axis = plt.subplots(1, num_epochs)
figure.subplots_adjust(top=0.8)
figure.set_figwidth(num_epochs)
figure.set_figheight(1)


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
        axis[col].set_title(f"ep. {epoch}")
        axis[col].set_ylim(-5, 5)
        axis[col].plot(values_history[-1][2])

        gauss_distribution = dff_simulator.initial_values[0] + dff_simulator.initial_values[1]
        gauss_distribution_shifted = gauss_distribution - tf.ones(gauss_distribution.shape)*3
        axis[col].plot(gauss_distribution_shifted)


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
