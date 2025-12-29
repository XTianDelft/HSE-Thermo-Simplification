import pickle
from multiprocessing import Pool
import pygfunction as gt
from GHEtool import *
import time

from heat_profile import *

T_g = 10     # undisturbed ground temperature
k_g = 2.8   # ground thermal conductivity
Cp = 2.8e6  # ground volumetric heat capacity in J/m3K
Rb = 0.1125
r_b = 0.125 #Borehole radius

# min and max fluid temperatures
Tf_max = 16
Tf_min = 0

max_tilt = 45
max_nb = 20

# number of processes that simultaneously calculate the lengths
# for different tilt angles and number of boreholes
nr_of_processes = 12
high_res = False
plot_heat_profile = True

ground = GroundConstantTemperature(k_g, T_g, Cp)

# profile_root = 'profiles/'
profile_root = 'profiles/tuinzicht/'
profiles_filenames = [
    # 'profile-test.xlsx',
    # 'profile-test2.xlsx',
    # 'profile-test3.xlsx',
    # 'profile-test-capped.xlsx',

    '71.xlsx',
    # '69.xlsx',
    # '67.xlsx',
]

results_root = 'results/'
results_name = '-'.join(map(lambda s: s.split('.')[0], profiles_filenames)) + ('' if high_res else '-LOWRES')


total_heat_profile = np.zeros(8760, dtype=np.float64)
print('Summing profiles:')
for profile_fn in profiles_filenames:
    heat_profile = read_hourly_profile(profile_root + profile_fn)
    print(f'\t{profile_fn.split("/")[-1]}\tenergy: {heat_profile.sum() / 1e6:.1f} MWh, max: {heat_profile.max() / 1e3:.1f} kW')
    assert len(heat_profile) == 8760, f'heat profile in file {profile_fn} has {len(heat_profile)} hours'
    total_heat_profile += heat_profile

print(f'total heat energy: {total_heat_profile.sum() / 1e6 :.1f} MWh')

if plot_heat_profile:
    print('plotting heat profile')
    plt.gcf().set_size_inches(10, 3)
    plt.plot(total_heat_profile/1e3, lw=.5)
    plt.xlim(0, 8760)
    plt.xlabel('time (hours)')
    plt.ylabel('building heat load (kW)')
    plt.tight_layout()
    plt.savefig(results_root + f'heat_profile-{results_name}.pdf')

load = HourlyBuildingLoad(total_heat_profile/1e3)


borefield = Borefield(load=load)

borefield.THRESHOLD_WARNING_SHALLOW_FIELD = 10
borefield.ground_data = ground
borefield.Rb = Rb
borefield.set_max_avg_fluid_temperature(Tf_max)
borefield.set_min_avg_fluid_temperature(Tf_min)


def size_bh_ring(degangle, Nb):
    # size a borefield using a custom field with pygfunction
    tilt = np.deg2rad(degangle)
    H = 30  # arbitrary initial norehole length (in meters)
    B = r_b/np.sin(np.pi/Nb)+.1  # radial Separation of borehole heads

    boreholes = [gt.boreholes.Borehole(H, 0, r_b, B*np.cos(phi), B*np.sin(phi), tilt=tilt, orientation=phi) for phi in np.linspace(0, 2*np.pi, Nb, endpoint=False)]
    borefield.set_borefield(gt.borefield.Borefield.from_boreholes(boreholes))

    length = borefield.size(L4_sizing=True) #each borehole length
    #print(f"Each BH is: {length} m")
    length_total = length*Nb

    # The constraints are not checked here. That is done at the visualization stage instead of at
    # the calculation stage, because it is so much faster it makes sense to separate these actions.
    return length_total

if high_res:
    angles = np.linspace(0, max_tilt, max_tilt+1)
    nb_range = np.arange(1, max_nb+1)
else:
    angles = np.linspace(0, max_tilt, 10)
    nb_range = np.arange(1, max_nb+1, 2)
nr_inputs = len(angles)*len(nb_range)

# Generate all the combinations for the inputs as a list of tuples
ang_vals, nb_vals = np.meshgrid(angles, nb_range)
inputs = list(zip(ang_vals.ravel(), nb_vals.ravel()))

# Wrapper function that unpacks the tuple containing the inputs
def do_sizing(inputs):
    degangle, Nb = inputs
    print(f'sizing {Nb} boreholes, {degangle}° tilted')
    total_length = size_bh_ring(degangle, Nb)
    print(f'done ({Nb}, {degangle}°): {total_length/Nb:.2f}')
    return total_length

starttime = time.time()
with Pool(nr_of_processes) as p:
    outputs = p.map(do_sizing, inputs)
print(f'took {time.time() - starttime:.3f} sec')

outputs = np.array(outputs)
lengths = outputs.reshape((len(angles), len(nb_range)))

saved_dict = {
    'angles': angles,
    'nb_range': nb_range,
    'lengths': lengths
}

results_filename = results_root + f"results-{results_name}.pkl"
print(f'writing results to {results_filename}')
with open(results_filename, "wb") as f_out:
    pickle.dump(saved_dict, f_out)
