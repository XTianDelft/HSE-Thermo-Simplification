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
r_b = 0.125

# min and max fluid temperatures
Tf_max = 16
Tf_min = 0


ground = GroundConstantTemperature(k_g, T_g, Cp)

profile_fn = "profiles/profile-test.xlsx"
heat_profile = read_hourly_profile(profile_fn)
load = HourlyBuildingLoad(heat_profile/1e3)

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

borefield.ground_data = ground
borefield.Rb = Rb
borefield.set_max_avg_fluid_temperature(Tf_max)
borefield.set_min_avg_fluid_temperature(Tf_min)

# custom field with pygfunction
tilt = np.deg2rad(45)
H = 30
B = .2
Nb = 9
boreholes = [gt.boreholes.Borehole(H, 0, r_b, B*np.cos(phi), B*np.sin(phi), tilt=tilt, orientation=phi) for phi in np.linspace(0, 2*np.pi, Nb, endpoint=False)]
gt_borefield = gt.borefield.Borefield.from_boreholes(boreholes)

borefield.set_borefield(gt_borefield)
# borefield.create_custom_dataset(options={'method': 'similarities'})

length = borefield.size(L4_sizing=True)
print(f"The borehole length (with {Nb} boreholes) is: {length} m")
print(f"{borefield.limiting_quadrant = }")

# print imbalance
print("The borefield imbalance is: ", borefield.load.imbalance,
        "kWh/y. (A negative imbalance means the the field is heat extraction dominated so it cools down year after year.)")  # print imbalance

Tf = borefield.results.Tf

# plot temperature profile for the calculated borehole length
borefield.print_temperature_profile(legend=True, plot_hourly=True)

# plot temperature profile for a fixed borehole length
# borefield.print_temperature_profile_fixed_length(length=75, legend=False)

# print gives the array of monthly temperatures for peak cooling without showing the plot
borefield.calculate_temperatures(length=90)
# print("Result array for cooling peaks")
# print(borefield.results.peak_injection)
