import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

r_b = 0.125
def calc_depth_and_radius(angle, Nb, length):
    radangle = np.deg2rad(angle)

    depth = np.cos(radangle)*length
    radius = np.sin(radangle)*length+(0 if Nb == 1 else r_b/np.sin(np.pi/Nb))

    # radius_violated =  > max_radius
    # depth_violated =  > max_depth
    # return radius_violated*2 + depth_violated
    # return not radius_violated and not depth_violated

    return depth, radius


#===Property Boundaries===
R_max = 15 #Max Radius in m
H_max = 40 #Max depth in m


profile_filename = 'profile-test.xlsx'
results_filename = f"results/results-{profile_filename.split('.')[0]}.pkl"
print(f'writing results to {results_filename}')
with open(results_filename, "rb") as f:
    saved_dict = pickle.load(f)

angles = saved_dict['angles']
nb_range = saved_dict['nb_range']
lengths = saved_dict['lengths']


depth = np.zeros((len(angles), len(nb_range)))
radius = np.zeros_like(depth)
for i in range(len(angles)):
    for j in range(len(nb_range)):
        depth[i,j], radius[i,j] = calc_depth_and_radius(angles[i], nb_range[j], lengths[i,j]/nb_range[j])


# --- Plotting code: ---

def contour_plot(arr, color, min_, max_=100, step=10):
    levels = np.arange(min_, min(max_, arr.max())+1, step)
    cs = plt.contour(arr, levels=levels, colors=color, origin='lower', extent=extent)
    plt.clabel(cs)


extent = (nb_range.min(), nb_range.max(), angles.min(), angles.max())

# plot constraints
# plt.imshow(lengths, norm=LogNorm(), origin='lower', extent=extent, aspect='auto')
# plt.colorbar()

contour_plot(depth, 'red', H_max)
contour_plot(radius, 'green', R_max, 100, 10)
contour_plot(lengths, 'blue', 100, 400, 20)



# plot lengths contour plot
# levels = np.arange(round(lengths.min(), -1), lengths.max(), 10)
# cs = plt.contour(lengths, levels=levels, origin='lower', extent=extent)
# plt.clabel(cs)

plt.xlabel('number of boreholes')
plt.ylabel('tilt angle $\\theta$ (deg)')

plt.tight_layout()
plt.show()

