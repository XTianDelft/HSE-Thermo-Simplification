import pickle
from multiprocessing import Pool
import pygfunction as gt
from GHEtool import *
import time

from heat_profile import *

starttime = time.time()

#--------------------Initial Conditions----------------------------
T_g = 10     # undisturbed ground temperature
k_g = 2.8   # ground thermal conductivity
Cp = 2.8e6  # ground volumetric heat capacity in J/m3K
Rb = 0.1125
r_b = 0.125 #Borehole radius
# min and max fluid temperatures
Tf_max = 16
Tf_min = 0

#==Done Setting, then Importing==

profile_filename = 'profile-test.xlsx'
heating_load = read_hourly_profile('profiles/' + profile_filename)/1e3
load = HourlyBuildingLoad(heating_load)
borefield = Borefield(load=load)
borefield.THRESHOLD_WARNING_SHALLOW_FIELD = 10
borefield.ground_data = GroundConstantTemperature(k_g, T_g, Cp)
borefield.Rb = Rb
borefield.set_max_avg_fluid_temperature(Tf_max)
borefield.set_min_avg_fluid_temperature(Tf_min)
#--------------------Initial Conditions----------------------------END

def size_bh_ring(degangle, Nb): # custom field with pygfunction
    tilt = np.deg2rad(degangle)
    H = 30 #Arbitrary Initial Borehole length (in meters) !!AUTO!!
    B = r_b/np.sin(np.pi/Nb) #Radial Separation of borehole heads !!AUTO!!
    #print("Borehole Radial Head Separation(Radius): ",B," [m]")

    boreholes = [gt.boreholes.Borehole(H, 0, r_b, B*np.cos(phi), B*np.sin(phi), tilt=tilt, orientation=phi) for phi in np.linspace(0, 2*np.pi, Nb, endpoint=False)]
    borefield.set_borefield(gt.borefield.Borefield.from_boreholes(boreholes))

    length = borefield.size(L4_sizing=True) #each borehole length
    #print(f"Each BH is: {length} m")
    length_total = length*Nb

    return length_total #check_constrains(degangle,length,R_max,H_max)


#-----------------variable Conditions-----------------ONLY FOR DEBUGGING

#for Nb in range(0,17):
    #for degangle in range(0,46,5):
        #print(Nb,degangle)

#Nb = 9 #Number of Boreholes
#degangle = 45 #degrees from vertical
#print ("Results[Length total BH, Fitment]: ", calculate(degangle, Nb))


angles = np.arange(0, 45+1, 1)
nb_range = np.arange(1, 16+1)
nr_inputs = len(angles)*len(nb_range)

inputs = np.array(np.meshgrid(angles, nb_range)).T.reshape(nr_inputs, 2)

def do_sizing(inputs):
    degangle, Nb = inputs
    print(f'sizing {Nb} boreholes, {degangle}° tilted')
    return size_bh_ring(degangle, Nb)

with Pool(12) as p:
    outputs = p.map(do_sizing, inputs)

outputs = np.array(outputs)
lengths = outputs.reshape((len(angles), len(nb_range)))

print(lengths)

saved_dict = {
    'angles': angles,
    'nb_range': nb_range,
    'lengths': lengths
}

results_filename = f"results/results-{profile_filename.split('.')[0]}.pkl"
print(f'writing results to {results_filename}')
with open(results_filename, "wb") as f_out:
    pickle.dump(saved_dict, f_out)
