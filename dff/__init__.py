import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

from dff.simulation.simulator import Simulator
from dff.visualization import *
