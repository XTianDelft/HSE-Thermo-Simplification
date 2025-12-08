import matplotlib.pyplot as plt
import numpy as np
import pygfunction as gt
import time
import pickle

# this is mostly from the example:
# Simulation of fluid temperatures in a field of multiple boreholes¶
# https://pygfunction.readthedocs.io/en/stable/examples/fluid_temperature_multiple_boreholes.html



T_g = 10

k_g = 1.2
rho = 1700
cp = 1800
alpha = k_g / (rho * cp)

# https://pygfunction.readthedocs.io/en/stable/modules/media.html
fluid = gt.media.Fluid('MPG', 20) # 20% propylene glycol in water at T=20°C
cp_f = fluid.cp

# alpha = 1e-6
print(f"α = {alpha:.2e}")

B = 0.707
D = 0

r_b = 0.125  # Borehole radius (m)

Hs = np.array([40, 42.4, 32.9, 41.8, 42, 42, 34.95, 43.5, 40])
H_mean = Hs.mean()
ts = H_mean ** 2 / (9 * alpha)
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

boreholes = [
    gt.boreholes.Borehole(Hs[i], D, r_b, *locs[i], tilt=phi[i], orientation=theta[i])
    for i in range(Nb)
]

for i in range(Nb):
    print(f"H: {Hs[i]:.2f} m,\ttilt: {np.rad2deg(phi[i]):.2f}°")

borefield = gt.borefield.Borefield.from_boreholes(boreholes)

print(f"H_mean: {H_mean:.2f} m,\tsteady state time = {ts / (3600*24*365.25):.2f} years")

# borefield.visualize_field()
# plt.savefig('setup.pdf')
# plt.close()


# very crude hourly energy in kWh/h
with open('./outputs/EnergyMeter_outputs.pkl', 'rb') as f:
    power_df = pickle.load(f)['power_df']


Nt = len(power_df)//4*4
Q_tot = power_df['Power_Energy_kWh'].to_numpy()
Q_tot = Q_tot[:Nt].reshape(-1, 4).sum(axis=1)

m_flow_network = power_df['Flow_Rate'].to_numpy()
m_flow_network = m_flow_network[:Nt].reshape(-1, 4).mean(axis=1)
m_flow_borehole = m_flow_network / Nb

inlet_temps = power_df['Inlet_Temperature'].to_numpy()
inlet_temps = inlet_temps[:Nt].reshape(-1, 4).mean(axis=1)

outlet_temps = power_df['Return_Temperature'].to_numpy()
outlet_temps = outlet_temps[:Nt].reshape(-1, 4).mean(axis=1)



# --- NETWORK DEFINITION ---

# 1. Define Pipe & Grout Properties (Standard HDPE assumptions)
r_out = 0.020  # Pipe outer radius (m) - approx 40mm diameter
r_in = 0.017  # Pipe inner radius (m)
D_s = 0.06  # Shank spacing (distance between pipe centers in m)
k_p = 0.4  # Pipe thermal conductivity (W/m.K)
k_s = 2.0  # Grout thermal conductivity (W/m.K)
epsilon = 1.0e-6  # Pipe roughness (m)


pipes = []
for bh in boreholes:
    # Generate pipe positions inside the borehole
    pos = gt.pipes.SingleUTube.positions(bh.r_b, D_s)

    # Create the pipe object
    pipe = gt.pipes.SingleUTube(
        pos, r_in, r_out, bh, k_s=k_s, k_g=k_g, k_p=k_p
    )
    pipes.append(pipe)

# 3. Create the Network
network = gt.networks.Network(
    boreholes,
    pipes,
    is_parallel=True,# We assume boreholes in parallel
    m_flow_network=m_flow_network[0],  # Initial flow estimate
    cp_f=cp_f
)

# === NETWORK DEFINITION ===


dt = 3600
tmax = len(Q_tot) * 3600
LoadAgg = gt.load_aggregation.ClaessonJaved(dt, tmax)
time_req = LoadAgg.get_times_for_simulation()

# time_arr = gt.utilities.time_geometric(3600, 20 * 8766 * 3600, 8)
gfunc = gt.gfunction.gFunction(
    borefield,
    alpha,
    time=time_req,
    boundary_condition="UBWT",
    method="similarities",
    options={'disp': True}
)
# gfunc.visualize_g_function()
# plt.savefig('gfunc.pdf')

LoadAgg.initialize(gfunc.gFunc / (2 * np.pi * k_g))

T_b = np.zeros(Nt)
T_f_in = np.zeros(Nt)
T_f_out = np.zeros(Nt)

for i in range(Nt):
    # Increment time step by (1)
    LoadAgg.next_time_step(i*3600)

    Q_i = Q_tot[i]

    # Apply current load (in watts per meter of borehole)
    Q_b = Q_i/Nb
    LoadAgg.set_current_load(Q_b/H_mean)

    # Evaluate borehole wall temperature
    deltaT_b = LoadAgg.temporal_superposition()
    T_b[i] = T_g - deltaT_b

    # Evaluate inlet fluid temperature (all boreholes are the same)
    T_f_in[i] = network.get_network_inlet_temperature(
            Q_tot[i], T_b[i], m_flow_network[i], cp_f, nSegments=1)

    # Evaluate outlet fluid temperature
    T_f_out[i] = network.get_network_outlet_temperature(
            T_f_in[i],  T_b[i], m_flow_network[i], cp_f, nSegments=1)


# Configure figure and axes
fig = gt.utilities._initialize_figure()

ax1 = fig.add_subplot(211)
# Axis labels
ax1.set_xlabel(r'Time [hours]')
ax1.set_ylabel(r'Total heat extraction rate [W]')
gt.utilities._format_axes(ax1)

# Plot heat extraction rates
hours = np.arange(1, Nt+1) * dt / 3600.
ax1.plot(hours, Q_tot)

ax2 = fig.add_subplot(212)
# Axis labels
ax2.set_xlabel(r'Time [hours]')
ax2.set_ylabel(r'Temperature [degC]')
gt.utilities._format_axes(ax2)

# Plot temperatures
ax2.plot(hours, T_b, label='Borehole wall')
ax2.plot(hours, T_f_out, '-.',
            label='Outlet, double U-tube (parallel)')
ax2.legend()

# Adjust to plot window
plt.tight_layout()
