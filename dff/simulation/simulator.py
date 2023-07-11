import threading
import time
import logging
from enum import Enum

import dff.simulation.steps.gauss_input
from dff.simulation.util import compute_positional_grid
from dff.simulation.weight_patterns import compute_weight_pattern_tensor, weight_pattern_config_from_dfpy_weight_pattern
from dfpy.activation_function import Sigmoid, Identity

from dff.simulation.simulator_graph import create_unrolled_simulation_call, create_rolled_simulation_call, \
    create_unrolled_simulation_call_with_history, create_impromptu_simulation_call_with_history
from dff.simulation.simulator_graph import simulate_unrolled_time_steps

logger = logging.getLogger(__name__)

from math import ceil, floor
from multiprocessing import Queue

import tensorflow as tf
from tensorflow import Tensor

from dfpy.connection import SynapticConnection
from dfpy.neural_structure import NeuralStructure
from dfpy.steps import *
from dfpy.steps import Step
import dfpy.shared

from dff.simulation import steps


def simulator_thread_target(simulator):
    while True:
        if simulator.async_max_time is not None:
            if simulator.async_simulation_running:
                simulator.async_max_time_lock.acquire()
                max_time = simulator.async_max_time
                simulator.async_max_time_lock.release()
                simulator.simulate(max_time)
            else:
                break
        else:
            simulator.simulate()


class SimulationCallMode(Enum):
    unrolled=1
    rolled=2


class SimulationCallType(Enum):
    largest=1
    single=2


class Simulator:
    """A simulator simulates the state of an architecture numerically over time. The TensorFlow-based simulator uses
    TensorFlow (https://www.tensorflow.org) as a backend for the computations.
    """

    def __init__(self, neural_structure: NeuralStructure = None,
                 time_step_duration: float=20.0, record_values=False,
                 queue: Queue=None, record_time_points_of_interest: bool=False,
                 debug_steps: list=[]):
        """Creates a TensorFlow simulator.

        :param neural_structure: the architecture to be simulated
        :param time_step_duration: duration of a time step (milliseconds)
        :param record_values: whether to record the state of the architecture across all time steps
        :param queue: optional queue into which the values of each time step will be put
        :param record_time_points_of_interest: whether to record time points of interest (e.g., when a peak
        forms)
        :param debug_steps: steps for which to print debug output
        """

        if neural_structure is None:
            neural_structure = dfpy.shared.get_default_neural_structure()
        self._neural_structure = neural_structure
        self._time_step_duration = tf.constant(time_step_duration, tf.float32)
        self._record_values = record_values

        self._queue = queue
        self._record_time_points_of_interest = record_time_points_of_interest
        self._debug_steps = debug_steps

        self._initial_values = None
        self._values = None
        self._recorded_values = []
        self._recorded_time_points_of_interest = []
        self._thread = None
        self._async_simulation_running = False
        self._async_max_time = 0
        self._async_max_time_lock = threading.Lock()
        self._values_lock = threading.Lock()

        self._time_step = tf.Variable(0, dtype=tf.int32)

        self._constants = {}
        self._variables = {}
        self._time_and_variable_invariant_tensors = {}
        self._time_invariant_variable_variant_tensors = {}

        self.prepare_constants_and_variables()
        self.prepare_time_and_variable_invariant_tensors()
        self.prepare_connection_kernel_weights_tensors()
        self.prepare_time_invariant_variable_variant_tensors()
        self.prepare_initial_values()
        self.reset_time()
        self._simulation_calls_with_unrolled_time_steps = {}
        self._rolled_simulation_call = None

        # Register property observers on steps
        for step in self._neural_structure.steps:
            step.register_observer(self._handle_modified_step)

        self._neural_structure.register_add_step_observer(self._handle_add_step)
        self._neural_structure.register_add_connection_observer(self._handle_add_connection)

    def _handle_modified_step(self, step, changed_param):
        raise RuntimeError("_handle_modified_step in simulator.py must be adapted to optimizations")
        step_index = self._neural_structure.steps.index(step)

        new_value = getattr(step, changed_param)
        if changed_param in self._variables[step]:
            self._variables[step][changed_param].assign(new_value)
        elif changed_param == "interaction_kernel" and isinstance(step, Field):
            weight_pattern_config = weight_pattern_config_from_dfpy_weight_pattern(new_value, step.domain(), step.shape())
            old_weight_pattern_config = self._variables[step]["interaction_kernel_weight_pattern_config"]
            if weight_pattern_config["type"] == "GaussWeightPattern":
                old_weight_pattern_config["sigmas"].assign(weight_pattern_config["sigmas"])
                old_weight_pattern_config["height"].assign(weight_pattern_config["height"])
            else:
                raise RuntimeError("Cannot replace kernel by " + weight_pattern_config["type"] + " because this feature is not implemented yet. Please talk to daniel.sabinasz@ini.rub.de.")
            #raise RuntimeError("Cannot replace the interaction kernel after building the computational graph. Please update the interaction kernel parameters instead.")
            #self._variables[step]["interaction_kernel_weight_pattern_config"] = \
            #    weight_pattern_config_from_dfpy_weight_pattern(new_value, step.domain(), step.shape())
            #self._simulation_calls_with_unrolled_time_steps = {}
            #self._rolled_simulation_call = None
        else:
            raise RuntimeError(f"Unrecognized changed_param '{changed_param}'")

        self._initial_values[step_index].assign(self.compute_initial_value_for_step(step_index, step))

        self._time_invariant_variable_variant_tensors[step] =\
            self.compute_time_invariant_variable_variant_tensors_for_step(step)
        self._values[step_index].assign(self.compute_initial_value_for_step(step_index, step))

    def _handle_add_step(self, step):
        raise RuntimeError("_handle_add_step in simulator.py must be adapted to optimizations")
        self.prepare_constants_and_variables_for_step(step)
        self.prepare_time_and_variable_invariant_tensors_for_step(step)
        self.prepare_connection_kernel_weights_tensors()
        step_index = self._neural_structure.steps.index(step)
        self._time_invariant_variable_variant_tensors[step] = \
            self.compute_time_invariant_variable_variant_tensors_for_step(step)
        initial_value = tf.Variable(self.compute_initial_value_for_step(step_index, step))
        self._values.append(initial_value)
        self._initial_values.append(initial_value)
        step.register_observer(self._handle_modified_step)
        self._simulation_calls_with_unrolled_time_steps = {}
        self._rolled_simulation_call = None

    def _handle_add_connection(self, connection):
        self.prepare_connection_kernel_weights_tensors()
        self._simulation_calls_with_unrolled_time_steps = {}
        self._rolled_simulation_call = None

    def prepare_constants_and_variables(self):
        steps = self._neural_structure.steps
        logger.debug(f"Preparing constants and variables for {len(steps)} steps")
        before = time.time()
        for i in range(0, len(steps)):
            step = steps[i]
            try:
                self.prepare_constants_and_variables_for_step(step)
            except Exception as e:
                raise RuntimeError(f"Error trying to prepare constants and variables for step {step.name}") from e
        logger.debug(f"Done preparing constants and variables after {time.time()-before} seconds")

    def prepare_constants_and_variables_for_step(self, step):
        #
        # TimedBoost
        #
        if isinstance(step, TimedBoost):
            constants = steps.timed_boost.timed_boost_prepare_constants(step)
            variables = {}

        #
        # Boost
        #
        elif isinstance(step, Boost):
            constants = steps.boost.boost_prepare_constants(step)
            variables = {}

        #
        # GaussInput
        #
        elif isinstance(step, GaussInput):
            constants = steps.gauss_input.gauss_input_prepare_constants(step)
            variables = steps.gauss_input.gauss_input_prepare_variables(step)

        #
        # CustomInput
        #
        elif isinstance(step, CustomInput):
            constants = {}
            variables = steps.custom_input.custom_input_prepare_variables(step)

        #
        # TimedCustomInput
        #
        elif isinstance(step, TimedCustomInput):
            constants = {}
            variables = steps.timed_custom_input.timed_custom_input_prepare_variables(step)

        #
        # TimedGate
        #
        elif isinstance(step, TimedGate):
            constants = steps.timed_gate.timed_gate_prepare_constants(step)
            variables = {}

        #
        # NoiseInput
        #
        elif isinstance(step, NoiseInput):
            constants = steps.noise_input.noise_input_prepare_constants(step)
            variables = {}

        #
        # Image
        #
        elif isinstance(step, Image):
            constants = steps.image.image_prepare_constants(step)
            variables = {}

        #
        # NeuralField
        #
        elif isinstance(step, Field):
            constants = steps.field.field_prepare_constants(step)
            variables = steps.field.field_prepare_variables(step)

        #
        # NeuralNode
        #
        elif isinstance(step, Node):
            constants = steps.node.node_prepare_constants(step)
            variables = steps.node.node_prepare_variables(step)

        #
        # Scalar
        #
        elif isinstance(step, Scalar):
            constants = {}
            variables = steps.scalar.scalar_prepare_variables(step)

        #
        # ScalarMultiplication
        #
        elif isinstance(step, ScalarMultiplication):
            constants = steps.scalar_multiplication.scalar_multiplication_prepare_constants(
                step
            )
            variables = steps.scalar_multiplication.scalar_multiplication_prepare_variables(
                step
            )

        else:
            raise RuntimeError(f"Unrecognized step type f{type(step)}")

        self._constants[step] = constants
        self._variables[step] = variables

    def prepare_time_and_variable_invariant_tensors(self):
        steps = self._neural_structure.steps
        logger.info(f"Preparing time-and-variable-invariant tensors for {len(steps)} steps...")
        before = time.time()
        for i in range(0, len(steps)):
            step = steps[i]

            try:
                self.prepare_time_and_variable_invariant_tensors_for_step(step)
            except Exception as e:
                raise RuntimeError(f"Error trying to prepare time- and variable-invariant tensors for step {step.name}") from e

        logger.info(f"Done preparing time-and-variable-invariant tensors after {time.time()-before} seconds")

    def prepare_time_and_variable_invariant_tensors_for_step(self, step):
        constants = self._constants[step]
        variables = self._variables[step]

        #
        # TimedBoost
        #
        if isinstance(step, TimedBoost):
            time_and_variable_invariant_tensors = {}

        #
        # Boost
        #
        elif isinstance(step, Boost):
            time_and_variable_invariant_tensors = {}

        #
        # GaussInput
        #
        elif isinstance(step, GaussInput):
            time_and_variable_invariant_tensors = steps.gauss_input\
                .gauss_input_prepare_time_and_variable_invariant_tensors(
                    constants["shape"],
                    constants["domain"]
                )

        #
        # CustomInput
        #
        elif isinstance(step, CustomInput):
            time_and_variable_invariant_tensors = {}

        #
        # TimedCustomInput
        #
        elif isinstance(step, TimedCustomInput):
            time_and_variable_invariant_tensors = {}

        #
        # TimedGate
        #
        elif isinstance(step, TimedGate):
            time_and_variable_invariant_tensors = {}

        #
        # Image
        #
        elif isinstance(step, Image):
            time_and_variable_invariant_tensors = {}

        #
        # NeuralField
        #
        elif isinstance(step, Field):
            # Should take ~0.026
            time_and_variable_invariant_tensors = steps.field\
                .field_prepare_time_and_variable_invariant_tensors(
                    step,
                    constants["shape"],
                    constants["domain"]
                )

        #
        # NeuralNode
        #
        elif isinstance(step, Node):
            time_and_variable_invariant_tensors = {}

        #
        # Scalar
        #
        elif isinstance(step, Scalar):
            time_and_variable_invariant_tensors = []

        #
        # ScalarMultiplication
        #
        elif isinstance(step, ScalarMultiplication):
            time_and_variable_invariant_tensors = []

        else:
            time_and_variable_invariant_tensors = []
            #raise RuntimeError(f"Unrecognized step: '{type(step)}'")

        self._time_and_variable_invariant_tensors[step] = time_and_variable_invariant_tensors

    def compute_initial_value_for_step(self, step_index, step, time_invariant_variable_variant_tensors = None):
        if time_invariant_variable_variant_tensors is None:
            time_invariant_variable_variant_tensors = self._time_invariant_variable_variant_tensors
        constants = self._constants[step]
        variables = self._variables[step]
        time_and_variable_invariant_tensors = self._time_and_variable_invariant_tensors[step]
        time_invariant_variable_variant_tensors = time_invariant_variable_variant_tensors[step]

        #
        # TimedBoost
        #
        if isinstance(step, TimedBoost):
            initial_value = steps.timed_boost.timed_boost_time_step(constants["values"], 0.0)

        #
        # Boost
        #
        if isinstance(step, Boost):
            initial_value = steps.boost.boost_time_step(constants["value"])

        #
        # GaussInput
        #
        if isinstance(step, GaussInput):
            initial_value = time_invariant_variable_variant_tensors["gauss_input_tensor"]
            #initial_value = steps.gauss_input.gauss_input_time_step(variables["height"],
            #                                                       variables["mean"],
            #                                                       variables["sigmas"],
            #                                                       time_and_variable_invariant_tensors["positional_grid"])

        #
        # CustomInput
        #
        if isinstance(step, CustomInput):
            initial_value = variables["pattern"]

        #
        # TimedCustomInput
        #
        if isinstance(step, TimedCustomInput):
            initial_value = variables["timed_custom_input"][0]

        #
        # TimedGate
        #
        if isinstance(step, TimedGate):
            initial_value = tf.zeros(step.shape())

        #
        # Image
        #
        if isinstance(step, Image):
            initial_value = constants["image_tensor"]

        #
        # NeuralField
        #
        if isinstance(step, Field):
            #initial_value = time_invariant_variable_variant_tensors[0] TODO performance
            initial_value = tf.ones(tuple([int(x) for x in constants["shape"]])) * variables["resting_level"]

        #
        # NeuralNode
        #
        if isinstance(step, Node):
            initial_value = variables["resting_level"]

        #
        # Scalar
        #
        if isinstance(step, Scalar):
            initial_value = variables["value"]

        #
        # ScalarMultiplication
        #
        if isinstance(step, ScalarMultiplication):
            initial_value = tf.zeros(tuple([int(x) for x in constants["shape"]]))

        #
        # NoiseInput
        #
        if isinstance(step, NoiseInput):
            initial_value = steps.noise_input.noise_input_time_step(self._time_step_duration, step._shape,
                                                                    step._strength)

        return initial_value

    #def reset_step_to_initial_value(self, step):
    #    self._values.write(self.neural_structure.steps.index(step), self.compute_initial_value_for_step(step))

    def prepare_time_invariant_variable_variant_tensors(self):
        logger.info(f"Preparing time-invariant variable-variant tensors for "
                    f"{len(self._neural_structure.steps)} steps...")
        before = time.time()
        tensors = self.compute_time_invariant_variable_variant_tensors()
        logger.info(f"Done preparing time-invariant variable-variant tensors after {time.time() - before} seconds")
        self._time_invariant_variable_variant_tensors = tensors

    def compute_time_invariant_variable_variant_tensors(self):
        tensors = {}
        for i in range(0, len(self._neural_structure.steps)):
            step = self._neural_structure.steps[i]
            tensors_for_step = self.compute_time_invariant_variable_variant_tensors_for_step(step)
            tensors[step] = tensors_for_step
        return tensors

    def compute_time_invariant_variable_variant_tensors_for_step(self, step):
        variables = self._variables[step]
        constants = self._constants[step]
        time_and_variable_invariant_tensors = self._time_and_variable_invariant_tensors[step]
        if isinstance(step, Field):
            tensors = {
                "resting_level_tensor": tf.ones(tuple([int(x) for x in constants["shape"]])) * variables["resting_level"],
                "lateral_interaction_weight_pattern_tensor": compute_weight_pattern_tensor(variables["interaction_kernel_weight_pattern_config"],
                                                                                           time_and_variable_invariant_tensors["interaction_kernel_positional_grid"])
            }
        elif isinstance(step, GaussInput):
            tensors = dff.simulation.steps.gauss_input.gauss_input_prepare_time_invariant_variable_variant_tensors(
                variables["mean"], variables["sigmas"], variables["height"],
                time_and_variable_invariant_tensors["positional_grid"])
        else:
            tensors = {}

        return tensors

    def update_initial_values(self, time_invariant_variable_variant_tensors=None):
        if time_invariant_variable_variant_tensors == None:
            time_invariant_variable_variant_tensors = self._time_invariant_variable_variant_tensors
        steps = self._neural_structure.steps
        for i in range(0, len(steps)):
            step = steps[i]
            initial_value = self.compute_initial_value_for_step(i, step, time_invariant_variable_variant_tensors)
            self._initial_values[i] = initial_value
        return self._initial_values

    def compute_initial_values(self, time_invariant_variable_variant_tensors):
        steps = self._neural_structure.steps
        initial_values = []
        for i in range(0, len(steps)):
            step = steps[i]
            initial_value = self.compute_initial_value_for_step(i, step, time_invariant_variable_variant_tensors)
            initial_values.append(initial_value)
        return initial_values

    def prepare_initial_values(self):
        self._initial_values = self.compute_initial_values(self._time_invariant_variable_variant_tensors)

    @property
    def initial_values(self):
        return self._initial_values

    def reset_time(self):
        self._values = []
        self._recorded_values = []
        self._recorded_time_points_of_interest = []
        self._async_max_time = 0
        self._time_step.assign(0)

        steps = self._neural_structure.steps
        for i in range(0, len(steps)):
            step = steps[i]

            # If values should be recorded, add a list for time step 0
            if self._record_values:
                self._recorded_values.append([])

            initial_value = self._initial_values[i]
            value = initial_value
            self._values.append(value)

            if self._record_values:
                self._recorded_values[i].append(initial_value)

        if self._queue is not None:
            self._queue.put([value.numpy() for value in self._values])

    @property
    def neural_structure(self):
        return self._neural_structure

    def prepare_connection_kernel_weights_tensors(self):
        # Transform everything into TensorFlow tensors

        self._connection_kernel_weights = {}
        self._connection_kernel_weight_pattern_configs = {}
        self._connection_kernel_positional_grids = {}
        self._connection_pointwise_weights = {}
        self._connection_contraction_weights = {}

        for i in range(len(self._neural_structure.steps)):
            step = self._neural_structure.steps[i]

            connection_kernel_weights= []
            connection_kernel_weight_pattern_configs = []
            connection_kernel_positional_grids = []
            connection_pointwise_weights = []
            connection_contraction_weights = []
            connections_into_step = self._neural_structure.connections_into_steps[i]
            for j in range(len(connections_into_step)):
                connection = connections_into_step[j]
                input_step_index = connection.input_step_index

                if isinstance(connection, SynapticConnection):

                    if connection.kernel_weights is not None:
                        domain = step.domain()
                        shape = step.shape()

                        lower = domain[0][0]
                        upper = domain[0][1]
                        rng = upper - lower
                        kernel_domain = [[-rng / 2, rng / 2]]
                        kernel_positional_grid = compute_positional_grid(shape, kernel_domain)

                        rng = connection.kernel_weights.range()
                        if rng is not None:
                            w = shape[0]
                            kernel_positional_grid = kernel_positional_grid[
                                                                 w // 2 - rng[1]:w // 2 + rng[0] + 1]

                        # TODO handle scalar case differently
                        kernel_weight_pattern_config = weight_pattern_config_from_dfpy_weight_pattern(connection.kernel_weights, domain, shape)
                        kernel_weights = compute_weight_pattern_tensor(
                            kernel_weight_pattern_config,
                            kernel_positional_grid
                        )
                    else:
                        kernel_weights = None
                        kernel_weight_pattern_config = None
                        kernel_positional_grid = None

                    if connection.pointwise_weights is not None:
                        if isinstance(step, Node):
                            pointwise_weights = tf.convert_to_tensor(connection.pointwise_weights)
                        else:
                            domain = step.domain()
                            shape = step.shape()
                            pointwise_weights = compute_weight_pattern_tensor(
                                weight_pattern_config_from_dfpy_weight_pattern(connection.pointwise_weights, domain,
                                                                               shape),
                                self._time_and_variable_invariant_tensors[step]["positional_grid"]
                            )
                    else:
                        pointwise_weights = None

                else:
                    if connection.kernel_weights is not None:
                        domain = step.domain()
                        shape = step.shape()

                        lower = domain[0][0]
                        upper = domain[0][1]
                        rng = upper - lower
                        kernel_domain = [[-rng / 2, rng / 2]]
                        kernel_positional_grid = compute_positional_grid(shape, kernel_domain)

                        # TODO handle scalar case differently
                        kernel_weight_pattern_config = weight_pattern_config_from_dfpy_weight_pattern(connection.kernel_weights, domain, shape)
                        kernel_weights = compute_weight_pattern_tensor(
                            kernel_weight_pattern_config,
                            kernel_positional_grid
                        )
                    else:
                        kernel_weights = None
                        kernel_weight_pattern_config = None
                        kernel_positional_grid = None

                    pointwise_weights = None

                connection_kernel_weight_pattern_configs.append(kernel_weight_pattern_config)
                connection_kernel_weights.append(kernel_weights)
                connection_kernel_positional_grids.append(kernel_positional_grid)
                connection_pointwise_weights.append(pointwise_weights)
                if connection.contraction_weights is not None:
                    contraction_weights = tf.convert_to_tensor(connection.contraction_weights)
                else:
                    contraction_weights = None
                connection_contraction_weights.append(contraction_weights)

            self._connection_kernel_weights[step] = connection_kernel_weights
            self._connection_kernel_weight_pattern_configs[step] = connection_kernel_weight_pattern_configs
            self._connection_kernel_positional_grids[step] = connection_kernel_positional_grids
            self._connection_pointwise_weights[step] = connection_pointwise_weights
            self._connection_contraction_weights[step] = connection_contraction_weights

    def simulate_time_step(self):
        """Simulates one time step.
        """
        return self.simulate_time_steps(1)

    def simulate_until(self, time: float, mode: str = None, in_multiples_of: int = None):
        if mode == None:
            mode = self._default_simulation_call_type
        if type(time) == int:
            time = float(time)
        current_time = self.get_time_as_tensor()
        duration_to_simulate = time - current_time
        num_time_steps = ceil(duration_to_simulate / self._time_step_duration)
        return self.simulate_time_steps(num_time_steps, mode, in_multiples_of)

    def simulate_for(self, duration: float, mode = None, in_multiples_of = None):
        if mode == None:
            mode = self._default_simulation_call_type
        num_time_steps = ceil(duration / self._time_step_duration)
        return self.simulate_time_steps(num_time_steps, mode, in_multiples_of)

    def get_unrolled_simulation_call(self, num_time_steps):
        if num_time_steps not in self._simulation_calls_with_unrolled_time_steps:
            #self._simulation_calls_with_unrolled_time_steps[num_time_steps] = \
            #    create_unrolled_simulation_call(self, num_time_steps, self._time_step_duration)
            self._simulation_calls_with_unrolled_time_steps[num_time_steps] = True
            new_graph = True
        else:
            new_graph = False
        return new_graph

    def get_unrolled_simulation_call_with_history(self, num_time_steps):
        return create_unrolled_simulation_call_with_history(self, num_time_steps, self._time_step_duration)

    def get_impromptu_simulation_call_with_history(self, num_time_steps):
        return create_impromptu_simulation_call_with_history(self, num_time_steps, self._time_step_duration)

    def get_rolled_simulation_call(self):
        if self._rolled_simulation_call is None:
            self._rolled_simulation_call = create_rolled_simulation_call(self, self._time_step_duration)
            new_graph = True
        else:
            new_graph = False
        return self._rolled_simulation_call, new_graph

    def get_largest_suitable_unrolled_simulation_call(self, max_num_time_steps):
        largest_num_time_steps_per_call = 0
        for num_time_steps_per_call in self._simulation_calls_with_unrolled_time_steps:
            if num_time_steps_per_call <= max_num_time_steps and num_time_steps_per_call > largest_num_time_steps_per_call:
                largest_num_time_steps_per_call = num_time_steps_per_call
                simulation_call = self._simulation_calls_with_unrolled_time_steps[largest_num_time_steps_per_call]

        if largest_num_time_steps_per_call == 0:
            new_graph = True
            simulation_call = create_unrolled_simulation_call(self, max_num_time_steps,
                                                              self._time_step_duration)
            largest_num_time_steps_per_call = max_num_time_steps
        else:
            new_graph = False

        return simulation_call, largest_num_time_steps_per_call, new_graph

    def simulate_time_steps(self, num_time_steps: int, in_multiples_of: int = None):
        """Simulates the specified number of time steps.

        :param num_time_steps: the number of time steps to simulate
        :param mode: simulation mode
        """
        logger.debug("simulate_time_steps " + str(num_time_steps))

        if in_multiples_of is not None:

            new_graph = self.get_unrolled_simulation_call(in_multiples_of)
            before_simulating = time.time()
            trace_duration = 0
            num_main_calls = floor(num_time_steps / in_multiples_of)
            if num_main_calls > 0:
                logger.info(
                    f"Running a simulation call for {in_multiples_of} time steps {num_main_calls} times.")
            for i in range(num_main_calls):
                if new_graph:
                    logger.info(f"Tracing simulation call with {in_multiples_of} time steps...")
                    before = time.time()
                self._values = simulate_unrolled_time_steps(self, in_multiples_of,
                                                            self.get_time_as_tensor() + i * in_multiples_of * self._time_step_duration,
                                                            self._time_step_duration, self._values)
                if new_graph:
                    trace_duration = time.time() - before
                    logger.info("Done tracing after " + str(trace_duration) + " seconds")
                    new_graph = False
            logger.info("Done simulating after " + str(time.time()-before_simulating-trace_duration) + " seconds")

        else:
            new_graph = self.get_unrolled_simulation_call(1)
            if num_time_steps > 0:
                before_simulating = time.time()
                trace_duration = 0
                for i in range(num_time_steps):
                    if new_graph:
                        logger.info(f"Tracing simulation call with 1 time step...")
                        before = time.time()
                    #self._values = simulation_call(self.get_time_as_tensor() + i*self._time_step_duration, self._values)
                    self._values = simulate_unrolled_time_steps(self, 1, self.get_time_as_tensor() + i*self._time_step_duration, self._time_step_duration, self._values)
                    if new_graph:
                        trace_duration = time.time()-before
                        logger.info("Done tracing after " + str(trace_duration) + " seconds")
                        new_graph = False
                if trace_duration == 0:
                    logger.info("Done simulating after " + str(time.time()-before_simulating-trace_duration) + " seconds")
            else:
                raise RuntimeError("Trying to simulate 0 time steps. This is probably a mistake.")

        # Replace values and increment time step
        self._values_lock.acquire()
        self._time_step.assign(self._time_step + num_time_steps)
        self._values_lock.release()

        # Record values
        if self._record_values:
            for i in range(0, len(self._neural_structure.steps)):
                self._recorded_values[i].append(self._values[i])

        # Record time points of interest
        # TODO: Check why this is a performance bottleneck
        #if self._record_time_points_of_interest:
        #    for i in range(0, len(self._neural_structure.steps)):
        #        step = self._neural_structure.steps[i]
        #        if isinstance(step, )NeuralNode:
        #           if new_values[i] > 0 and self._values[i] <= 0:
        #                t_np = t.numpy()
        #                print("on", t_np, self._neural_structure.steps[i].name)
        #                if t_np not in self._recorded_time_points_of_interest:
        #                    self._recorded_time_points_of_interest.append(t_np)
        #            elif new_values[i] < 0 and self._values[i] >= 0:
        #                t_np = t.numpy()
        #                print("off", t_np, self._neural_structure.steps[i].name)


        # Print debug output for debug steps
        #if len(self._debug_steps) > 0:
        #    print("t = " + str(t))
        #    for step in self._debug_steps:
        #        print("- " + step.name)
        #        input_step_indices = self._neural_structure.input_steps_by_step_index[self._neural_structure.steps.index(step)]
        #        for input_step_index in input_step_indices:
        #            input_step = self._neural_structure.steps[input_step_index]
        #            print("  - " + input_step.name + ": " + str(self._values[input_step_index]))

        # Add values to queue
        if self._queue is not None:
            self._queue.put([value.numpy() for value in self._values])

        return self._values

    def simulate(self, max_time: int = None):
        """Simulates until the specified max time.

        :param max_time: maximum time until which to simulate (inclusive)
        """

        if max_time is not None:
            current_time = self.get_time()
            if current_time >= max_time:
                return
            num_time_steps = ceil((max_time-current_time) / self._time_step_duration.numpy())
            self.simulate_time_steps(num_time_steps)
        else:
            self.simulate_time_steps(1)

    def start_async_simulation(self):
        self._async_simulation_running = True
        if self._thread is None:
            self._thread = threading.Thread(target=simulator_thread_target, args=(self,))
            self._thread.start()

    def stop_async_simulation(self):
        self._async_simulation_running = False
        self._thread = None

    @property
    def async_max_time_lock(self):
        return self._async_max_time_lock

    @property
    def values_lock(self):
        return self._values_lock

    @property
    def async_max_time(self):
        return self._async_max_time

    @async_max_time.setter
    def async_max_time(self, async_max_time):
        self._async_max_time = async_max_time

    @property
    def async_simulation_running(self):
        return self._async_simulation_running

    def get_tensorflow_node(self, step) -> object:
        """Returns the tensorflow node for the specified step.

        :param Step step: the step whose tensorflow node should be returned
        :return tensorflow node of the specified step
        """
        return self._tensorflow_nodes[self._neural_structure.steps.index(step)]

    def get_value(self, step: Step):
        """Get the current value for the specified step.

        :param step: the step whose value should be returned
        :return: the current value of the step
        """
        return self._values[self._neural_structure.steps.index(step)]

    def get_recorded_values(self) -> list:
        """Get recorded values.

        :return: the recorded values indexed by step and time step
        """
        return self._recorded_values

    def get_recorded_values_for_step(self, step) -> list:
        """Get recorded values for the specified step.

        :param Step or int step: the step whose recorded values should be returned (either step object or index)
        :return: the recorded values indexed by time step
        """
        if isinstance(step, int):
            return self._recorded_values[step]
        return self._recorded_values[self._neural_structure.steps.index(step)]

    def get_recorded_values_at_time_step(self, time_step: int) -> list:
        """Get recorded values for the specified time step.

        :param time_step: the time step
        :return: the recorded values, indexed by step index
        """
        ret = []
        for i in range(len(self._neural_structure.steps)):
            ret.append(self._recorded_values[i][time_step])
        return ret

    def get_recorded_values_at_time(self, time: int) -> list:
        """Get recorded values for the specified time.

        :param time: the time
        :return: the recorded values, indexed by step index
        """
        time_step = time / self._time_step_duration.numpy()
        return self.get_recorded_values_at_time_step(time_step)

    def get_recorded_value_at_time_step(self, step, time_step: int):
        """Get recorded value for the specified step at the specified time step.

        :param Step or int step: the step whose recorded values should be returned (either step object or index)
        :param time_step: time step
        :return: recorded value
        """
        return self._recorded_values[self._neural_structure.steps.index(step)][time_step]

    def get_recorded_value_at_time(self, step, time: int):
        """Get recorded value for the specified step at the specified time.

        :param Step or int step: the step whose recorded values should be returned (either step object or index)
        :param time: the time
        :return: recorded value
        """

        time_step = int(time / self._time_step_duration.numpy())
        if not self._record_values:
            return self._values[self._neural_structure.steps.index(step)]
        recorded_values = self._recorded_values[self._neural_structure.steps.index(step)]
        if len(recorded_values) < time_step+1:
            return None
        return recorded_values[time_step]

    def get_recorded_time_points_of_interest(self) -> list:
        """Get recorded time points of interest.

        :return: recorded time points of interest
        """
        return self._recorded_time_points_of_interest

    def get_time_step(self) -> Tensor:
        """Get the current time step.

        :return: the current time step
        """
        return self._time_step

    def get_time(self) -> int:
        """Get the current time.

        :return: the current time
        """
        return self.get_time_as_tensor().numpy()

    def get_time_as_tensor(self) -> Tensor:
        """Get the current time as a tensor.

        :return: the current time
        """
        return tf.multiply(tf.cast(self._time_step, tf.float32), self._time_step_duration)

    @property
    def time_step_duration(self) -> float:
        """Get the time step duration.

        :return: the time step duration
        """
        return self._time_step_duration

    @property
    def variables(self) -> dict:
        """Get trainable variables.

        :return: the trainable variables
        """
        return self._variables

    @property
    def record_values(self):
        return self._record_values


"""@tf.function
def get_input_sum_by_step_index(steps, connections_into_steps, values):
    logger.debug("trace get_input_sum_by_step_index")
    # Create a new empty TensorArray that will hold the input sum by step
    input_sum_by_step_index = tf.TensorArray(dtype=tf.float32, size=len(values), infer_shape=False, clear_after_read=False)

    # Loop over steps
    for i in range(0, len(steps)):
        # If there are connections into the step,
        if len(connections_into_steps[i]) > 0:
            # Get the first incoming connection
            connection = connections_into_steps[i][0]
            input = values[connection.input_step_index]
            if type(connection) == SynapticConnection:
                input = tf.math.sigmoid(tf.multiply(connection.activation_function.beta, input))
                input = tf.multiply(connection.synaptic_weight_pattern, input)
            input_sum = input

            # Iterate all other incoming connections
            for j in range(1, len(connections_into_steps[i])):
                # Get the connection
                connection = connections_into_steps[i][j]
                # Get the input step value
                input = values[connection.input_step_index]
                # Synaptic connections get a special treatment:
                # The input is sigmoided and multiplied with the synaptic weights
                if type(connection) == SynapticConnection:
                    input = tf.math.sigmoid(tf.multiply(connection.activation_function.beta, input))
                    input = tf.multiply(connection.synaptic_weight_pattern, input)

                input_sum = tf.add(input_sum, input)

            input_sum_by_step_index = input_sum_by_step_index.write(i, input_sum)

    return input_sum_by_step_index"""
