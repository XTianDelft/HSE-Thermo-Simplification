import pickle
import numpy as np
import matplotlib.pyplot as plt

precalc_fp = 'results/precalc_borefields-k_g=2.80-Cp=2.80e+06-T_g=10.0-LOWRES.pkl'

print(f'loading {precalc_fp}')
with open(precalc_fp, "rb") as f:
    saved_dict = pickle.load(f)

angles = saved_dict['angles']
nb_range = saved_dict['nb_range']
borefields = saved_dict['borefields']

ang_idx = 9
nb_idx = 9

print(f'showing g-functions for tilt: {angles[ang_idx]} and Nb: {nb_range[nb_idx]}')
bh = borefields[ang_idx, nb_idx]

gfuncs = bh.custom_gfunction.gvalues_array
bh_lens = bh.custom_gfunction.borehole_length_array
time_arr = bh.custom_gfunction.time_array

for len_idx in range(0, len(bh_lens)):
    bh_len = bh_lens[len_idx]
    plt.plot(np.log(time_arr), gfuncs[len_idx], label=bh_len)

plt.legend(ncol=4)
plt.tight_layout()
plt.show()
