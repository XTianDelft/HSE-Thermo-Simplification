import time
import numpy as np
import pygfunction as gt
from GHEtool import *

from heat_profile import *

T_g = 10     # undisturbed ground temperature
k_g = 2.8   # ground thermal conductivity
Cp = 2.8e6  # ground volumetric heat capacity in J/m3K
Rb = 0.1125
r_b = 0.125 #Borehole radius

alpha = k_g/Cp

# min and max fluid temperatures
Tf_max = 16
Tf_min = 0

# custom field with pygfunction
tilt = np.deg2rad(13)
H = 40 #Arbitrary Initial Borehole length (in meters)
Nb = 8 #Number of Boreholes
B = (r_b+.05)/np.sin(np.pi/Nb) #Radial Separation of borehole heads
# print("Borehole Radial Head Separation(Radius): ",B," [m]")


ground = GroundConstantTemperature(k_g, T_g, Cp)
heat_profile = read_hourly_profile('profiles/profile-test.xlsx')
load = HourlyBuildingLoad(heat_profile/1e3)


borefield = Borefield()

borefield.THRESHOLD_WARNING_SHALLOW_FIELD = 10
borefield.ground_data = ground
borefield.Rb = Rb
borefield.set_max_avg_fluid_temperature(Tf_max)
borefield.set_min_avg_fluid_temperature(Tf_min)

phis = np.linspace(0, 2*np.pi, Nb, endpoint=False)
gt_borefield = gt.borefield.Borefield(H, 0, r_b, B*np.cos(phis), B*np.sin(phis), tilt, phis)
borefield.set_borefield(gt_borefield)
borefield.set_options_gfunction_calculation(options={'linear_threshold': 10*3600})

bh_lengths = np.arange(10, 80+1, 20)
borefield.create_custom_dataset(borehole_length_array=bh_lengths, options={'method': 'similarities'})

start = time.time()

borefield.set_load(load)
length = borefield.size(L4_sizing=True)
print(time.time() - start, length)
