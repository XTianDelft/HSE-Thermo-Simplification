import pickle
import numpy as np
import matplotlib.pyplot as plt

precalc_fp = 'results/precalc_borefields-k_g=2.18-Cp=2.93e+06-T_g=10.0-HIGHRES.pkl'

print(f'loading {precalc_fp}')
with open(precalc_fp, "rb") as f:
    saved_dict = pickle.load(f)

angles = saved_dict['angles']
nb_range = saved_dict['nb_range']
borefields = saved_dict['borefields']

# check negative g-functions
for ang_idx in range(len(angles)):
    for nb_idx in range(len(nb_range)):
        bh = borefields[ang_idx, nb_idx]
        gfuncs = bh.custom_gfunction.gvalues_array
        bh_lens = bh.custom_gfunction.borehole_length_array
        if (gfuncs < 0).any():
            ids_len = np.where((gfuncs < 0).any(axis=1))[0]
            print(f'negative gfunc for {angles[ang_idx]:.1f}°, {nb_range[nb_idx]} at lengths: {bh_lens[ids_len]}')


ang_idx = 9
nb_idx = 9

# for ang_idx in range(len(angles)):
#     for nb_idx in range(len(nb_range)):
print(f'showing g-functions for tilt: {angles[ang_idx]} and Nb: {nb_range[nb_idx]}')
bh = borefields[ang_idx, nb_idx]

gfuncs = bh.custom_gfunction.gvalues_array
bh_lens = bh.custom_gfunction.borehole_length_array
time_arr = bh.custom_gfunction.time_array

len_min = bh_lens.min()
len_max = bh_lens.max()

for len_idx in range(0, len(bh_lens)):
    bh_len = bh_lens[len_idx]
    plt.semilogx(time_arr, gfuncs[len_idx], color=plt.cm.jet(((bh_len - len_min) / (len_max - len_min))**.3), label=bh_len)

plt.legend(ncol=4, fontsize=7)
plt.gcf().set_size_inches(8, 5)
plt.tight_layout()
plt.show()
