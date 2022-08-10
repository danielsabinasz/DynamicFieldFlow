import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.DEBUG)

from dff.simulation.simulator import Simulator
