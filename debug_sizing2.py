import pygfunction as gt
import numpy as np
import matplotlib.pyplot as plt
from GHEtool import *

degangle = 27
Nb = 11
tilt = np.deg2rad(degangle)
H = 40  # initial height (does not matter)
r_b = 0.125 # Borehole radius
B = (r_b+.05)/np.sin(np.pi/Nb)  # radial Separation of borehole heads

T_g = 10     # undisturbed ground temperature
k_g = 2.8   # ground thermal conductivity
Cp = 2.8e6  # ground volumetric heat capacity in J/m3K
Rb = 0.1125

Tf_max = 16
Tf_min = 0

ground = GroundConstantTemperature(k_g, T_g, Cp)

gfunc_options = {
    'method': 'similarities',
    'linear_threshold': 5*3600
}

borefield = Borefield()

borefield.THRESHOLD_WARNING_SHALLOW_FIELD = 10
borefield.ground_data = ground
borefield.Rb = Rb
borefield.set_max_avg_fluid_temperature(Tf_max)
borefield.set_min_avg_fluid_temperature(Tf_min)

phis = np.linspace(0, 2*np.pi, Nb, endpoint=False)
gt_borefield = gt.borefield.Borefield(H, 0, r_b, B*np.cos(phis), B*np.sin(phis), tilt, phis)



borefield.set_borefield(gt_borefield)

borefield.create_custom_dataset(borehole_length_array=[5, 7], options=gfunc_options)

gfuncs = borefield.custom_gfunction.gvalues_array
bh_lens = borefield.custom_gfunction.borehole_length_array
time_arr = borefield.custom_gfunction.time_array

if (gfuncs < 0).any():
    print("ERROR: Negative g-functions")
    ids_len = np.where((gfuncs < 0).any(axis=1))[0]
    print(f'negative gfunc for {degangle}°, {Nb} at lengths: {bh_lens[ids_len]}')

for i in range(len(gfuncs)):
    plt.plot(gfuncs[i], label=bh_lens[i])

plt.legend()
plt.show()
