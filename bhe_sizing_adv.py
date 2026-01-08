import pickle
from tqdm.contrib.concurrent import process_map
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
do_parallel = True
nr_of_processes = 4
high_res = True
plot_heat_profile = True

sizing_kwargs = {
    'atol': 0.2,
    'rtol': 0.01,
}

gfunc_options = {'linear_threshold': 5*3600}

ground = GroundConstantTemperature(k_g, T_g, Cp)

# profile_root = 'profiles/'
profile_root = 'profiles/tuinzicht/'
profile_filenames = [
    # 'profile-test.xlsx',
    # 'profile-test2.xlsx',
    # 'profile-test3.xlsx',
    # 'profile-test-capped.xlsx',

    'A71.xlsx',
]

def size_bh_ring(degangle, Nb):
        # size a borefield using a custom field with pygfunction
        tilt = np.deg2rad(degangle)
        H = 30  # arbitrary initial norehole length (in meters)
        B = (r_b+.05)/np.sin(np.pi/Nb)  # radial Separation of borehole heads

        phis = np.linspace(0, 2*np.pi, Nb, endpoint=False)
        gt_borefield = gt.borefield.Borefield(H, 0, r_b, B*np.cos(phis), B*np.sin(phis), tilt, phis)
        borefield.set_borefield(gt_borefield)
        borefield.set_options_gfunction_calculation(options=gfunc_options)

        try:
            length = borefield.size(L4_sizing=True, **sizing_kwargs) #each borehole length
        except VariableClasses.BaseClass.MaximumNumberOfIterations:
            print(f'ERROR: tilt = {degangle:.2f}°, Nb = {Nb} did not converge')
            length = np.float64('NaN')

        #print(f"Each BH is: {length} m")
        length_total = length*Nb

        # The constraints are not checked here. That is done at the visualization stage instead of at
        # the calculation stage, because it is so much faster it makes sense to separate these actions.
        return length_total

# Wrapper function for parallelizing that unpacks the tuple containing the inputs
def do_sizing(inputs):
    degangle, Nb = inputs
    # print(f'sizing {Nb} boreholes, {degangle}° tilted')
    total_length = size_bh_ring(degangle, Nb)
    # print(f'done ({Nb}, {degangle}°): {total_length/Nb:.2f}')
    return total_length


def advanced_sizing(profile_filenames, profile_root, case_name=None, living_space=None):
    results_root = 'results/'
    if case_name == None:
        results_name = '-'.join(map(lambda s: s.split('.')[0], profile_filenames))
    else:
        results_name = case_name
    results_name += '' if high_res else '-LOWRES'

    total_heat_profile = np.zeros(8760, dtype=np.float64)
    print(f'Summing profiles for {results_name}:')
    for profile_fn in profile_filenames:
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
        # display house properties on plot
        if living_space is not None:
            plt.title(((case_name + ': ') if case_name else '') + str(living_space) + ' m²', y=1., pad=-14)
            fs = (lambda kW: kW*1e3/living_space, lambda Wpm2: Wpm2*living_space/1e3)
            plt.gca().secondary_yaxis('right', functions=fs).set_ylabel('normalized heat load (W/m²)')
        plt.tight_layout()
        plt.savefig(results_root + f'heat_profile-{results_name}.pdf')

    load = HourlyBuildingLoad(total_heat_profile/1e3)

    global borefield
    borefield = Borefield(load=load)
    borefield.THRESHOLD_WARNING_SHALLOW_FIELD = 10
    borefield.ground_data = ground
    borefield.Rb = Rb
    borefield.set_max_avg_fluid_temperature(Tf_max)
    borefield.set_min_avg_fluid_temperature(Tf_min)


    if high_res:
        angles = np.linspace(0, max_tilt, max_tilt+1)
        nb_range = np.arange(1, max_nb+1)
    else:
        angles = np.linspace(0, max_tilt, 10)
        nb_range = np.arange(1, max_nb+1, 1)
    nr_inputs = len(angles)*len(nb_range)

    # Generate all the combinations for the inputs as a list of tuples
    ang_vals, nb_vals = np.meshgrid(angles, nb_range, indexing='ij')
    inputs = list(zip(ang_vals.ravel(), nb_vals.ravel()))

    starttime = time.time()

    if do_parallel:
        outputs = process_map(do_sizing, inputs, max_workers=nr_of_processes)
    else:
        outputs = np.zeros_like(inputs, dtype=np.float64)
        for i, inp in enumerate(inputs):
            print(f'{i+1}/{len(inputs)}')
            outputs[i] = do_sizing(inp)

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


if __name__ == '__main__':
    advanced_sizing(profile_filenames, profile_root)
