import logging

logging.basicConfig(level=logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

from dff.simulation.simulator import Simulator
from dff.visualization import *
