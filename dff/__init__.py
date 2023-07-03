import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

from dff.simulation.simulator import Simulator
from dff.visualization import *
