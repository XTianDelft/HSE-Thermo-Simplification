import numpy as np
from pygfunction.boreholes import rectangle_field
from pygfunction.gfunction import gFunction
from pygfunction.load_aggregation import ClaessonJaved
import matplotlib.pyplot as plt

field = rectangle_field(10,7,2,2,150,2,0.2)
time = ClaessonJaved(3600, 3600*8760*20).get_times_for_simulation()
options = {'linear_threshold': 3600.}
g_func = gFunction(field, .8e-6, time, options=options)
g_func.visualize_g_function()
plt.show()
