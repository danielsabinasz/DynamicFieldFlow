from dff.loss_criteria import reward_exact_activation_value, reward_positive_activation, reward_negative_activation
from dfpy import *
from dff import *
import tensorflow as tf
import matplotlib.pyplot as plt

t_max=int(600)
time_step_duration=int(20)
num_time_steps=int(t_max/time_step_duration)
num_epochs = 100
num_repetitions = 100


for diff in np.arange(0.0, 2.00, 0.01, dtype=np.float32):
    initialize_architecture()

    gauss1 = GaussInput(
        dimensions=[Dimension(0, 50, 51)],
        mean=[10.0],
        height=3.2,
        sigmas=[3.0]
    )
    gauss1.trainable = True

    gauss2 = GaussInput(
        dimensions=[Dimension(0, 50, 51)],
        mean=[40.0],
        height=3.0,
        sigmas=[3.0]
    )
    gauss2.trainable = True

    field = Field(
        dimensions=[Dimension(0, 50, 51)],
        resting_level=-5.0,
        interaction_kernel=SumWeightPattern([
            GaussWeightPattern(height=2.5, sigmas=(3.0,)),
            GaussWeightPattern(height=0.0, sigmas=(5.0,))
        ]),
        noise_strength=0.0,
        global_inhibition=-0.9,
    )
    field.trainable = True

    connect(gauss1, field)
    connect(gauss2, field)

    timed_boost = TimedCustomInput(
        timed_custom_input=[0.0]*15 + [3.0]*15
    )
    connect(timed_boost, field)

    noise_input = NoiseInput(
        dimensions=[Dimension(0, 50, 51)],
        strength=10.0
    )
    connect(noise_input, field)

    dff_simulator = Simulator(time_step_duration=time_step_duration)
    ns= dfpy.shared.get_default_neural_structure()


    simulation_call = dff_simulator.get_unrolled_simulation_call_with_history(num_time_steps)
    time_invariant_variable_variant_tensors = dff_simulator.compute_time_invariant_variable_variant_tensors()
    initial_values = dff_simulator.update_initial_values(time_invariant_variable_variant_tensors)
    values_history = simulation_call(0, initial_values)

    print("")

    desired_offset = 0.0

    @tf.function
    def loss():
        time_invariant_variable_variant_tensors = dff_simulator.compute_time_invariant_variable_variant_tensors()
        initial_values = dff_simulator.update_initial_values(time_invariant_variable_variant_tensors)

        # Perform simulation
        values_history = simulation_call(0, initial_values)

        before_boost_snapshot_field = values_history[15][2]
        before_boost_activation_left = before_boost_snapshot_field[10]
        before_boost_activation_right = before_boost_snapshot_field[40]

        before_boost_loss = (reward_negative_activation(before_boost_activation_left, 5) +
                             reward_negative_activation(before_boost_activation_right, 5))
        left_advantage_loss = tf.math.square((before_boost_activation_left - before_boost_activation_right)-diff)

        final_snapshot_field = values_history[-1][2]
        final_activation_left = final_snapshot_field[10]
        final_activation_right = final_snapshot_field[40]

        cf = 1/5.0
        peak_left = reward_positive_activation(final_activation_left, cf) + 2*reward_negative_activation(final_activation_right, cf)
        peak_right = 2*reward_negative_activation(final_activation_left, cf) + reward_positive_activation(final_activation_right, cf)
        selective_peak = tf.math.minimum(peak_right, peak_left)


        total_loss = before_boost_loss + selective_peak + 3*left_advantage_loss
        tf.print(before_boost_loss, selective_peak, left_advantage_loss)

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
        gauss2_height = dff_simulator.variables[gauss2]["height"]
        resting_level = dff_simulator.variables[field]["resting_level"]
        local_exc_height = dff_simulator.variables[field]["interaction_kernel_weight_pattern_config"]["summands"][0]["height"]
        local_exc_sigma = dff_simulator.variables[field]["interaction_kernel_weight_pattern_config"]["summands"][0]["sigmas"]
        global_inhibition = dff_simulator.variables[field]["global_inhibition"]
        #connection_kernel_height = dff_simulator._connection_kernel_weight_pattern_configs[field][0]["height"]

        print(f"gauss1_height={gauss1_height.numpy()}")
        print(f"gauss2_height={gauss2_height.numpy()}")
        print(f"resting_level={resting_level.numpy()}")
        print(f"local_exc_height={local_exc_height.numpy()}")
        print(f"local_exc_sigma={local_exc_sigma.numpy()}")
        print(f"global_inhibition={global_inhibition.numpy()}")
        #print(f"connection_kernel_height={connection_kernel_height.numpy()}")

        trainable_vars = [resting_level, local_exc_height, local_exc_sigma]
        gradients = tape.gradient(ls, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))


    time_invariant_variable_variant_tensors = dff_simulator.compute_time_invariant_variable_variant_tensors()
    initial_values = dff_simulator.update_initial_values(time_invariant_variable_variant_tensors)
    values_history = simulation_call(0, initial_values)
    plt.plot(values_history[15][2])
    plt.show()
    plt.plot(values_history[-1][2])
    plt.show()

    n_left = 0
    n_right = 0
    n_both = 0
    n_none = 0

    for rep in range(num_repetitions):
        time_invariant_variable_variant_tensors = dff_simulator.compute_time_invariant_variable_variant_tensors()
        initial_values = dff_simulator.update_initial_values(time_invariant_variable_variant_tensors)
        values_history = simulation_call(0, initial_values)
        final_snapshot_field = values_history[-1][2]
        final_activation_left = final_snapshot_field[10]
        final_activation_right = final_snapshot_field[40]
        left = final_activation_left > 0
        right = final_activation_right > 0

        if left:
            n_left += 1
        if right:
            n_right += 1
        if left and right:
            n_both += 1
        if not left and not right:
            n_none += 1

            """time_course = [values_history[t][2].numpy()
                           for t in range(num_time_steps)]
    
            skip = 1
            figure, axis = plt.subplots(1, num_time_steps // skip + 1)
            figure.subplots_adjust(top=0.8)
            figure.set_figwidth(2 * num_time_steps // skip + 1)
            figure.set_figheight(2)
            figure.suptitle("Time course")
            for t in range(num_time_steps):
                if t % skip != 0:
                    continue
                axis[t // skip].grid()
                axis[t // skip].set_title(f"t{time_step_duration * t}")
                axis[t // skip].set_ylim(-15, 15)
                axis[t // skip].plot(time_course[t])
            plt.show()
            plt.close()"""

    print(str(diff) + " " + str(n_left) + " " + str(n_right) + " " + str(n_both) + " " + str(n_none) + "\n")
    with open('out.txt', 'a') as f:
        f.write(str(diff) + " " + str(n_left) + " " + str(n_right) + " " + str(n_both) + " " + str(n_none) + "\n")
    print()
