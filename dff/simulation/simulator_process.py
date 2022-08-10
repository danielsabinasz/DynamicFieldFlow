from multiprocessing.context import Process
from multiprocessing.queues import Queue

from dff.architecture import Architecture
from dff.simulation import Simulator


def simulator_process_target(architecture, queue, max_time):
    simulator = Simulator(architecture, queue=queue, record_time_points_of_interest=True,
                                    record_computation_duration=True)
    simulator.simulate(max_time)
    queue.put(0)


class SimulatorProcess(Process):
    """A process that executes a simulator.
    """

    def __init__(self, architecture: Architecture, values_queue: Queue, max_time: float = None):
        """Creates a SimulatorProcess.

        :param architecture: The architecture to simulate
        :param values_queue: A queue into which the calculated values should be inserted
        :param max_time: The maximum time until which to simulate (inclusive)
        """
        super().__init__(target=simulator_process_target, args=(architecture, values_queue, max_time))
        self._values_queue = values_queue

    def values_queue(self):
        return self._values_queue
