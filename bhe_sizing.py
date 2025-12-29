import pickle
import numpy as np
import matplotlib.pyplot as plt
import pygfunction as gt
from GHEtool import *

from heat_profile import *

T_g = 10     # undisturbed ground temperature
k_g = 2.8   # ground thermal conductivity
Cp = 2.8e6  # ground volumetric heat capacity in J/m3K
Rb = 0.1125
r_b = 0.125 #Borehole radius

# min and max fluid temperatures
Tf_max = 16
Tf_min = 0

# custom field with pygfunction
tilt = np.deg2rad(45)
H = 30 #Arbitrary Initial Borehole length (in meters)
Nb = 20 #Number of Boreholes
B = r_b/np.sin(np.pi/Nb) #Radial Separation of borehole heads
# print("Borehole Radial Head Separation(Radius): ",B," [m]")


ground = GroundConstantTemperature(k_g, T_g, Cp)

# file_root = 'profiles/'
file_root = 'profiles/kennemerland/'
profiles_filenames = [
    # 'profile-test.xlsx',
    # 'profile-test2.xlsx',
    # 'profile-test3.xlsx',
    # 'profile-test-capped.xlsx',

    # '46.xlsx',
    # '44.xlsx',

    # '42.xlsx',
    # '40.xlsx',

    # '38.xlsx',
    # '36.xlsx',

    '34.xlsx',
    '32.xlsx',
    '30.xlsx',
    '28.xlsx',
    '26.xlsx',
    '24.xlsx',

]

total_heat_profile = np.zeros(8760, dtype=np.float64)
print('Summing profiles:')
for profile_fn in profiles_filenames:
    heat_profile = read_hourly_profile(file_root + profile_fn)
    print(f'\t{profile_fn.split("/")[-1]}\tenergy: {heat_profile.sum() / 1e6:.1f} MWh, max: {heat_profile.max() / 1e3:.1f} kW')
    assert len(heat_profile) == 8760, f'heat profile in file {profile_fn} has {len(heat_profile)} hours'
    total_heat_profile += heat_profile

print(f'total heat energy: {total_heat_profile.sum() / 1e6 :.1f} MWh')
load = HourlyBuildingLoad(total_heat_profile/1e3)

# with open('./HSE data/outputs/EnergyMeter_outputs.pkl', 'rb') as f:
#     aggregated_df = pickle.load(f)['aggregated_df']
# heat_profile_kW = aggregated_df['Power_KW'].to_numpy()[:8760]
# is_extracting = heat_profile_kW >= 0
# extraction_load = heat_profile_kW * is_extracting
# injection_load = -(heat_profile_kW * ~is_extracting)
# plt.plot(injection_load)
# plt.show()
# load = HourlyGeothermalLoad(extraction_load)



borefield = Borefield(load=load)

borefield.THRESHOLD_WARNING_SHALLOW_FIELD = 10
borefield.ground_data = ground
borefield.Rb = Rb
borefield.set_max_avg_fluid_temperature(Tf_max)
borefield.set_min_avg_fluid_temperature(Tf_min)

boreholes = [gt.boreholes.Borehole(H, 0, r_b, B*np.cos(phi), B*np.sin(phi), tilt=tilt, orientation=phi) for phi in np.linspace(0, 2*np.pi, Nb, endpoint=False)]
gt_borefield = gt.borefield.Borefield.from_boreholes(boreholes)

borefield.set_borefield(gt_borefield)
# borefield.create_custom_dataset(options={'method': 'similarities'})

length = borefield.size(L4_sizing=True) #each borehole length
depth = length * np.cos(tilt)
tip_radius = B + length * np.sin(tilt)

print("\n\nThe chosen layout:")
print(f"A ring of {Nb} boreholes (r={B:.2f} m), of length: {length:.1f} m (depth: {depth:.2f} m, bh tip radius: {tip_radius:.2f}) at an angle of {np.rad2deg(tilt):.2f}°")
print("\n")
print(f"{borefield.limiting_quadrant = }")

# print imbalance
print(f"The borefield imbalance is: {borefield.load.imbalance:.0f} kWh/y. (A negative imbalance means the the field is heat extraction dominated so it cools down year after year.)")  # print imbalance

Tf = borefield.results.Tf

# plot temperature profile for the calculated borehole length
plot_temp = False
if plot_temp:
    borefield.print_temperature_profile(legend=True, plot_hourly=True)
