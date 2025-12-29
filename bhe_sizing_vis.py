import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

r_b = 0.125
def calc_depth_and_radius(angle, Nb, length):
    radangle = np.deg2rad(angle)

    depth = np.cos(radangle)*length
    radius = np.sin(radangle)*length+(0 if Nb == 1 else r_b/np.sin(np.pi/Nb)+.1)

    # radius_violated =  > max_radius
    # depth_violated =  > max_depth
    # return radius_violated*2 + depth_violated
    # return not radius_violated and not depth_violated

    return depth, radius


R_max = 15  # max property radius in m
H_max = 40  # max depth in m

results_name = 'results-profile-test.pkl'
with open('results/' + results_name, "rb") as f:
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
ang_vals, nb_vals = np.meshgrid(angles, nb_range, indexing='ij')


contour_extent = (nb_range.min(), nb_range.max(), angles.min(), angles.max())

nb_step = nb_range[1] - nb_range[0]
ang_step = angles[1] - angles[0]
imshow_extent = (nb_range.min() - nb_step/2, nb_range.max() + nb_step/2,
                 angles.min() - ang_step/2, angles.max() + ang_step/2)

def contour_plot(arr, color, min_, max_=100, step=10):
    levels = np.arange(min_, min(max_, arr.max())+1, step)
    cs = plt.contour(arr, levels=levels, colors=color, origin='lower', extent=contour_extent)
    plt.clabel(cs)

def plot_all_contour():
    contour_plot(depth, 'red', H_max)
    contour_plot(radius, 'green', R_max, 100, 10)
    contour_plot(lengths, 'blue', 100, 400, 20)

def plot_binary_constraints():
    constraint_mask = (depth <= H_max) & (radius <= R_max)# & (ang_vals >= 10)
    valid_lengths = lengths.copy()
    valid_lengths[~constraint_mask] = np.float64('nan')

    plt.imshow(lengths, cmap='plasma', vmin=np.nanmin(valid_lengths), vmax=np.nanmax(valid_lengths), aspect='auto', origin='lower', extent=imshow_extent)
    plt.colorbar()

    len_contour_min = round(np.nanmin(valid_lengths), -1)
    contour_plot(valid_lengths, 'white', len_contour_min, len_contour_min+50, 10)

    # mark parts where constraints are not met red (depth) and orange (property radius)
    plt.contourf(depth, levels=[H_max, depth.max()+1], colors='#ff000080', origin='lower', extent=contour_extent)
    plt.contourf(radius, levels=[R_max, radius.max()], colors='#ff600080', origin='lower', extent=contour_extent)
    print(contour_extent)

# plot_all_contour()
plot_binary_constraints()

plt.xlabel('number of boreholes')
plt.xticks(nb_range)
plt.xlim(nb_range.min(), nb_range.max())
plt.ylabel('tilt angle $\\theta$ (deg)')
plt.ylim(angles.min(), angles.max())

plt.tight_layout()
plot_filename = f'results/config_space_plot-{results_name.split(".")[0]}.pdf'
plt.savefig(plot_filename)
plt.show()

