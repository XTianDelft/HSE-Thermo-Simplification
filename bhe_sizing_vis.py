import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as mpatches


# R_max = float('inf')
R_max = 25  # max property radius in m
H_max = 50  # max depth in m

results_name = 'results-Eilandstraat1.pkl'
results_root = 'results/'
results_fp = results_root + results_name

def visualize(results_name):
    results_fp = results_root + results_name
    print(f'loading {results_fp}')
    with open(results_fp, "rb") as f:
        saved_dict = pickle.load(f)

    angles = saved_dict['angles']
    nb_range = saved_dict['nb_range']
    lengths = saved_dict['lengths']

    per_meter_cost = 16
    per_bh_cost = 100
    def calc_cost(total_length, Nb):
        return total_length*per_meter_cost + Nb*per_bh_cost

    r_b = 0.125
    def calc_B(Nb):
        if Nb == 1:
            return 0
        else:
            return (r_b+.05)/np.sin(np.pi/Nb)

    def calc_depth_and_radius(angle, Nb, length):
        radangle = np.deg2rad(angle)
        depth = np.cos(radangle)*length
        radius = np.sin(radangle)*length+calc_B(Nb)

        return depth, radius

    depth = np.zeros((len(angles), len(nb_range)), dtype=np.float64)
    radius = np.zeros_like(depth)
    costs = np.zeros_like(depth)

    for i in range(len(angles)):
        for j in range(len(nb_range)):
            ang = angles[i]
            Nb = nb_range[j]
            length = lengths[i,j]
            depth[i,j], radius[i,j] = calc_depth_and_radius(ang, Nb, length/Nb)
            costs[i,j] = calc_cost(length, Nb)


    # --- Plotting code: ---
    ang_vals, nb_vals = np.meshgrid(angles, nb_range, indexing='ij')
    print('plotting for angles: ', angles)
    print('and Nbs: ', nb_range)
    print('\nlengths:')
    print(np.array2string(lengths, precision=1))
    print(f'depth: min, mean, max => {depth.min():.2f}, {depth.mean():.2f}, {depth.max():.2f}')
    print(f'radius: min, mean, max => {radius.min():.2f}, {radius.mean():.2f}, {radius.max():.2f}')

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
        plt.contourf(depth, levels=[H_max, np.nanmax(depth)+1], colors='#ff000080', origin='lower', extent=contour_extent)
        plt.contourf(radius, levels=[R_max, np.nanmax(radius)], colors='#ff600080', origin='lower', extent=contour_extent)

        print(contour_extent)

    def plot_all_pixels():
        # plot constraint violations
        colors = ['none', 'fuchsia', 'red', 'maroon']
        cmap = matplotlib.colors.ListedColormap(colors)
        im = ((depth > H_max) << 1) | (radius > R_max)
        # plot where which constraints are violated
        plt.imshow(im, cmap=cmap, vmin=0, vmax=3, aspect='auto', origin='lower', extent=imshow_extent)

        # plot costs
        constraint_mask = (depth <= H_max) & (radius <= R_max)# & (ang_vals >= 10)
        valid_costs = costs.copy()
        valid_costs[~constraint_mask] = np.float64('nan')
        plt.imshow(valid_costs, cmap='viridis', vmin=np.nanmin(valid_costs), vmax=np.nanmax(valid_costs), aspect='auto', origin='lower', extent=imshow_extent)
        plt.colorbar(label='Cost according sizing (€)')


        # mark minimum cost
        ang_idx, nb_idx = np.unravel_index(np.nanargmin(valid_costs), valid_costs.shape)
        best_nb = nb_range[nb_idx]
        best_ang = angles[ang_idx]
        best_len = lengths[ang_idx, nb_idx]
        best_cost = valid_costs[ang_idx, nb_idx]
        plt.scatter(best_nb, best_ang, c='white', edgecolors='black', label=f'min. cost: (€{per_meter_cost} $L$ + €{per_bh_cost})$N_b$ = €{best_cost:.0f}\n$L={best_len/best_nb:.1f}$ m at {best_ang:.0f}°, $N_b={best_nb}$')

        print('\nthe best configuration is as follows:')
        print(f'{Nb = :d} (B = {calc_B(best_nb):.2f} m), tilt = {best_ang:.1f}° =>\nlength = {best_len:.2f} m, depth = {depth[ang_idx, nb_idx]:.2f} m, radius = {radius[ang_idx, nb_idx]:.2f} m')

        # add constraint violations to the legend
        handles, labels = plt.gca().get_legend_handles_labels()

        legend_colors = colors[1:]
        legend_labels = [f'radius violated\n$r > {R_max}$ m',
                        f'depth violated\n$h > {H_max}$ m', 'both violated']
        new_handles = [mpatches.Patch(color=legend_colors[i], label=legend_labels[i]) for i in range(len(legend_colors))]

        new_handles.extend(handles)

        plt.legend(bbox_to_anchor=(-.1, 1.14), loc="upper left", ncols=len(new_handles), handles=new_handles)



    plot_all_pixels()

    # plot_all_contour()
    # plot_binary_constraints()

    plt.xlabel('number of boreholes')
    plt.xticks(nb_range)
    plt.xlim(nb_range.min(), nb_range.max())
    plt.ylabel('tilt angle $\\theta$ (deg)')
    plt.ylim(angles.min(), angles.max())

    plt.gcf().set_size_inches(8.4, 6)
    plt.tight_layout()
    plt.subplots_adjust(right=1)
    plot_filename = f'results/config_space_plot-{results_name.split(".")[0]}.pdf'
    plt.savefig(plot_filename)
    plt.savefig(plot_filename.replace('.pdf', '.png'), dpi=360)
    # plt.show()
    plt.clf()

if __name__ == '__main__':
    visualize(results_fp)
