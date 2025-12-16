import pickle
import numpy as np
import matplotlib.pyplot as plt
import pygfunction as gt
from GHEtool import *

from heat_profile import *

#--------------------Initial Conditions----------------------------
T_g = 10     # undisturbed ground temperature
k_g = 2.8   # ground thermal conductivity
Cp = 2.8e6  # ground volumetric heat capacity in J/m3K
Rb = 0.1125
r_b = 0.125 #Borehole radius
# min and max fluid temperatures
Tf_max = 16
Tf_min = 0
#Property Boundaries
R_max = 30 #Max Radius in m
H_max = 40 #Max depth in m

#==Done Setting, then Importing==

ground = GroundConstantTemperature(k_g, T_g, Cp)

profile_fn = "profiles/profile-test.xlsx"
heat_profile = read_hourly_profile(profile_fn)
load = HourlyBuildingLoad(heat_profile/1e3)

borefield = Borefield(load=load)

borefield.ground_data = ground
borefield.Rb = Rb
borefield.set_max_avg_fluid_temperature(Tf_max)
borefield.set_min_avg_fluid_temperature(Tf_min)
#--------------------Initial Conditions----------------------------END


def fitment(angle,length,max_radius,max_depth):
    fit = True
    radangle = np.deg2rad(angle)
    if np.sin(radangle)*length > max_radius:#radius check
        fit = False
        print("Too far horizontally!")
    elif np.cos(radangle)*length > max_depth:#depth check
        fit = False
        print("Too far vertically!")
    return fit

def calculate(degangle, Nb): # custom field with pygfunction

    tilt = np.deg2rad(degangle)
    H = 30 #Arbitrary Initial Borehole length (in meters) AUTO
    B = r_b/np.sin(np.pi/Nb) #Radial Separation of borehole heads AUTO
    print("Borehole Radial Head Separation(Radius): ",B," [m]")

    boreholes = [gt.boreholes.Borehole(H, 0, r_b, B*np.cos(phi), B*np.sin(phi), tilt=tilt, orientation=phi) for phi in np.linspace(0, 2*np.pi, Nb, endpoint=False)]
    borefield.set_borefield(gt.borefield.Borefield.from_boreholes(boreholes))

    length = borefield.size(L4_sizing=True) #each borehole length
    print(f"Each BH is: {length} m")

    return length


#-----------------variable Conditions-----------------
Nb = 9 #Number of Boreholes
degangle = 45 #degrees from vertical


print ("Length each Results: ", calculate(degangle, Nb))

print ("Fitment Results: ", fitment(0,45,R_max,H_max))