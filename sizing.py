import pygfunction as gt
import numpy as np
from GHEtool import *

k_g = 2.8   # ground thermal conductivity (W/mK)
T_g = 10    # initial/undisturbed ground temperature (deg C)
Cp = 2.8e6  # volumetric heat capacity of the ground (J/m3K)

Rb = 0.1125 # effective thermal resistance of the boreholes
r_b = 0.125 # grout radius

ground = GroundConstantTemperature(k_g, T_g, Cp)

peak_injection = np.zeros(12)
peak_extraction = np.array([160., 142, 102., 55., 0., 0., 0., 0., 40.4, 85., 119., 136.])  # Peak extraction in kW

monthly_load_extraction = np.array([46500.0, 44400.0, 37500.0, 29700.0, 19200.0, 0.0, 0.0, 0.0, 18300.0, 26100.0, 35100.0, 43200.0])  # in kWh
monthly_load_injection = np.zeros(12)

load = MonthlyGeothermalLoadAbsolute(monthly_load_extraction, monthly_load_injection, peak_extraction, peak_injection)

# create the borefield object
borefield = Borefield(load=load)

# set ground parameters
borefield.set_ground_parameters(ground)

# set the borehole equivalent resistance
borefield.Rb = Rb

# set temperature boundaries
borefield.set_max_avg_fluid_temperature(16)  # maximum temperature
borefield.set_min_avg_fluid_temperature(0)  # minimum temperature

# set a rectangular borefield
borefield_gt = gt.borefield.Borefield.rectangle_field(10, 12, 6, 6, 50, 0, r_b)
borefield.set_borefield(borefield_gt)

borefield.print_temperature_profile(legend=True)
