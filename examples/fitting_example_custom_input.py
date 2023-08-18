from dfpy import *
from dff import *
import tensorflow as tf
import matplotlib.pyplot as plt

t_max=int(300)
time_step_duration=int(20)
num_time_steps=int(t_max/time_step_duration)
num_epochs = 50

custom_input = CustomInput(pattern=[0.0]*10+[6.0]*10+[0.0]*31)
custom_input.assignable = True

field = Field(
    dimensions=[Dimension(0, 50, 51)],
    resting_level=-5.0,
    interaction_kernel=SumWeightPattern([
        GaussWeightPattern(height=1.0, sigmas=(3.0,)),
        GaussWeightPattern(height=-0.1, sigmas=(5.0,))
    ]),
    global_inhibition=-1.2,
)
field.trainable = True

connect(custom_input, field)

dff_simulator = Simulator(time_step_duration=time_step_duration)
ns= dfpy.shared.get_default_neural_structure()


simulation_call = dff_simulator.get_unrolled_simulation_call_with_history(num_time_steps)
time_invariant_variable_variant_tensors = dff_simulator.compute_time_invariant_variable_variant_tensors()
initial_values = dff_simulator.update_initial_values(time_invariant_variable_variant_tensors)


@tf.function
def loss():

    # Perform simulation
    values_history = simulation_call(0, initial_values)

    total_loss = -tf.reduce_sum(values_history[2])

    return total_loss


# Create an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.2)


losses = []
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        ls = loss()
    losses.append(ls)

    print(f"training epoch {epoch}")
    print(f"loss {ls}")
    gauss1_height = dff_simulator.variables[gauss1]["height"]
    connection_kernel_height = dff_simulator._connection_kernel_weight_pattern_configs[field][0]["height"]
    local_exc_height = dff_simulator.variables[field]["interaction_kernel_weight_pattern_config"]["summands"][0]["height"]
    print(f"gauss1_height={gauss1_height.numpy()}")
    print(f"connection_kernel_height={connection_kernel_height.numpy()}")
    print(f"local_exc_height={local_exc_height.numpy()}")

    trainable_vars = [local_exc_height]
    gradients = tape.gradient(ls, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))


time_invariant_variable_variant_tensors = dff_simulator.compute_time_invariant_variable_variant_tensors()
values_history = simulation_call(0, dff_simulator.initial_values)
plt.plot(values_history[-1][2])
plt.show()
