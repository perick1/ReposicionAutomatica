# Import modules
import numpy as np
import matplotlib.pyplot as plt
import random as rm

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)

np.random.seed(32)
rm.seed(12)
