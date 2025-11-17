import matplotlib.pyplot as plt
import numpy as np
import pygfunction as gt
import time

k_g = 1.2
rho = 1700
cp = 1800
# alpha = k_g / (rho * cp)
alpha = 1e-6
print(f'α = {alpha:.2e}')

B = 0.707
D = 0

r_b = 0.08  # Borehole radius (m)

Hs = np.array([40, 42.4, 32.9, 41.8, 42, 42, 34.95, 43.5, 40])
ts = Hs.mean() ** 2 / (9 * alpha)
dirs = np.array(
    [
        (0, 0, -1),
        (-0.3535533906, -0.3535533906, -0.8660254038),
        (0, -0.5877852523, -0.809016),
        (0.532625912, -0.532625912, -0.657737999),
        (0.4226182617, 0, -0.9063077870),
        (0.495952835, 0.495952835, -0.712784379),
        (-0.245168846, 0.657661762, -0.712301371),
        (-0.473661805, 0.473661805, -0.742488376),
        (-0.3420201433, 0, -0.9396926208),
    ]
)
locs = np.array(
    [
        (0, 0),
        (-0.707 * 0.7, -0.707 * 0.7),
        (0, -0.707 * 0.7),
        (0.707 * 0.7, -0.707 * 0.7),
        (0.707 * 0.7, 0),
        (0.707 * 0.7, 0.707 * 0.7),
        (0, 0.707 * 0.7),
        (-0.707 * 0.7, 0.707 * 0.7),
        (-0.707 * 0.7, 0),
    ]
)
Nb = len(dirs)
dx, dy, dz = dirs.T
dr = np.sqrt(dx**2 + dy**2)
phi = np.arctan2(dr, -dz)
theta = np.arctan2(dy, dx)

setup_bh = []
for i in range(Nb):
    setup_bh.append(
        gt.boreholes.Borehole(
            Hs[i], D, r_b, *locs[i], tilt=phi[i], orientation=theta[i]
        )
    )
    print(f'tilt: {np.rad2deg(phi[i]):.2f}')

tilt = 44.58/180*np.pi
boreholes_confs = {
    'comsol setup': setup_bh,
    'ring'+str(Nb): [
        gt.boreholes.Borehole(Hs.mean(), D, r_b, B/2*np.cos(phi), B/2*np.sin(phi), tilt=tilt, orientation=phi) for phi in np.linspace(0, 2*np.pi, Nb, endpoint=False)
    ],
    'ring'+str(Nb-1)+'+1': [
        gt.boreholes.Borehole(Hs.mean(), D, r_b, B/2*np.cos(phi), B/2*np.sin(phi), tilt=tilt, orientation=phi) for phi in np.linspace(0, 2*np.pi, Nb-1, endpoint=False)
    ] + [gt.boreholes.Borehole(Hs.mean(), D, r_b, 0, 0, tilt=0, orientation=0)],
}

for conf_name in boreholes_confs:
    boreholes = boreholes_confs[conf_name]
    borefield = gt.borefield.Borefield.from_boreholes(boreholes)

    H_sum = 0
    for bh in boreholes:
        H_sum += bh.H
    H_mean = H_sum/len(boreholes)
    ts = H_mean ** 2 / (9 * alpha)
    print(f'H_mean = {H_mean}, ts = {ts / (3600*24*365.25):.2f} years')


    # borefield.visualize_field()
    # plt.title(conf_name)
    # plt.savefig("setup.pdf")
    # plt.show()

    # time_arr = np.array([1, 6, 24*30, 24*365, 24*365*20])*3600
    # time_arr = np.concatenate([np.arange(1, 24)*3600, np.arange(1, 7)*3600*24, np.arange(1, 4)*3600*24*7, np.arange(1, 12)*3600*24*30.5, np.arange(1, 20)*3600*24*365.25])
    time_arr = gt.utilities.time_geometric(3600, 20 * 8766 * 3600, 25)

    begin = time.time()
    gfunc = gt.gfunction.gFunction(
        borefield, alpha, time=time_arr, boundary_condition="UBWT", method="similarities"
    )
    # print(f'that took {time.time()-begin:.2f} seconds')

    # gfunc.visualize_g_function()
    plt.plot(np.log(time_arr/ts), gfunc.gFunc, label=conf_name, marker='x')

plt.grid()
plt.legend()
plt.savefig("gfunc.pdf")
plt.show()
