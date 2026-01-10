import pickle
from tqdm.contrib.concurrent import process_map
import pygfunction as gt
from GHEtool import *
import time

from heat_profile import *

# T_g = 10     # undisturbed ground temperature
# k_g = 2.8   # ground thermal conductivity
# Cp = 2.8e6  # ground volumetric heat capacity in J/m3K
# ground = GroundConstantTemperature(k_g, T_g, Cp)
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
nr_of_processes = 12
high_res = False
plot_heat_profile = True
use_precalc = True


# profile_root = 'profiles/'
profile_root = 'profiles/tuinzicht/'
profile_filenames = [
    # 'profile-test.xlsx',
    # 'profile-test2.xlsx',
    # 'profile-test3.xlsx',
    # 'profile-test-capped.xlsx',

    'A71.xlsx',
]

results_root = 'results/'

def load_profiles_and_sum(profile_filenames, profile_root):
    total_heat_profile = np.zeros(8760, dtype=np.float64)
    for profile_fn in profile_filenames:
        heat_profile = read_hourly_profile(profile_root + profile_fn)
        print(f'\t{profile_fn.split("/")[-1]}\tenergy: {heat_profile.sum() / 1e6:.1f} MWh, max: {heat_profile.max() / 1e3:.1f} kW')
        assert len(heat_profile) == 8760, f'heat profile in file {profile_fn} has {len(heat_profile)} hours'
        total_heat_profile += heat_profile
    return total_heat_profile

def plot_case_heat_profile(case_heat_profile, plot_fn):
    print('plotting heat profile')
    plt.gcf().set_size_inches(10, 3)
    plt.plot(case_heat_profile/1e3, lw=.5)
    plt.xlim(0, 8760)
    plt.xlabel('time (hours)')
    plt.ylabel('building heat load (kW)')
    # display house properties on plot
    # if living_space is not None:
    #     plt.title(((case_name + ': ') if case_name else '') + str(living_space) + ' m²', y=1., pad=-14)
    #     fs = (lambda kW: kW*1e3/living_space, lambda Wpm2: Wpm2*living_space/1e3)
    #     plt.gca().secondary_yaxis('right', functions=fs).set_ylabel('normalized heat load (W/m²)')
    plt.tight_layout()
    plt.savefig(plot_fn)

def size_multiple_cases(cases, profile_root, results_root, precalc_filepath):
    # gather the case profiles
    cases_heat_profile = {}
    for case_name, fns in cases.items():
        fns_with_ext = list(map(lambda fn: fn + '.xlsx', fns))
        print(f'Loading and summing profiles for {case_name}:')
        total_heat_profile = load_profiles_and_sum(fns_with_ext, profile_root)
        cases_heat_profile[case_name] = total_heat_profile
        if plot_heat_profile:
            plot_case_heat_profile(total_heat_profile, results_root + f'heat_profile-{case_name}.pdf')

    # load precalculated borefields
    print(f'loading {precalc_filepath}')
    with open(precalc_filepath, "rb") as f:
        saved_dict = pickle.load(f)

    angles = saved_dict['angles']
    nb_range = saved_dict['nb_range']
    borefields = saved_dict['borefields']

    # size each case
    for case_name, case_heat_profile in cases_heat_profile.items():
        size_case_precalc(case_name, case_heat_profile, angles, nb_range, borefields)


def size_precalc(inputs):
    degangle, Nb = inputs

def size_case_precalc(case_name, case_heat_profile, angles, nb_range, borefields):
    print(f'Sizing {case_name}: total heat energy: {case_heat_profile.sum() / 1e6 :.1f} MWh')

    load = HourlyBuildingLoad(case_heat_profile/1e3)

    ang_len = len(angles)
    nb_len = len(nb_range)

    lengths = np.zeros((ang_len, len(nb_range)), dtype=np.float64)
    for ang_idx in range(ang_len):
        for nb_idx in range(nb_len):
            print(f'{ang_idx}/{ang_len} {nb_idx}/{nb_len}')

            if use_precalc:
                borefield = borefields[ang_idx, nb_idx]
            else:
                borefield = Borefield()
                borefield.THRESHOLD_WARNING_SHALLOW_FIELD = 10
                borefield.ground_data = ground
                borefield.Rb = Rb
                borefield.set_max_avg_fluid_temperature(Tf_max)
                borefield.set_min_avg_fluid_temperature(Tf_min)

                Nb = nb_range[nb_idx]
                H = 40
                B = (r_b+.05)/np.sin(np.pi/Nb)
                tilt = np.deg2rad(angles[ang_idx])

                phis = np.linspace(0, 2*np.pi, Nb, endpoint=False)
                gt_borefield = gt.borefield.Borefield(H, 0, r_b, B*np.cos(phis), B*np.sin(phis), tilt, phis)
                borefield.set_borefield(gt_borefield)
                gfunc_options = {
                    'method': 'similarities',
                    'linear_threshold': 5*3600
                }
                borefield.set_options_gfunction_calculation(options=gfunc_options)

            borefield.set_load(load)

            try:
                length = borefield.size(L4_sizing=True)
                length_total = length*nb_range[nb_idx]
                lengths[ang_idx, nb_idx] = length_total
            except:
                print(f'ERROR sizing {angles[ang_len]}°, {nb_range[nb_idx]} for {case_name}')
                lengths[ang_idx, nb_idx] = float('NaN')

    saved_dict = {
        'angles': angles,
        'nb_range': nb_range,
        'lengths': lengths
    }

    results_filename = results_root + f"results-{case_name}.pkl"
    print(f'writing results to {results_filename}')
    with open(results_filename, "wb") as f_out:
        pickle.dump(saved_dict, f_out)


if __name__ == '__main__':
    precalc_filepath = results_root + 'precalc_borefields-k_g=2.18-Cp=2.93e+06-T_g=10.0-LOWRES.pkl'

    case_name = 'profile-test'
    case_heat_profile = load_profiles_and_sum([f'{case_name}.xlsx'], 'profiles/')

    # load precalculated borefields
    print(f'loading {precalc_filepath}')
    with open(precalc_filepath, "rb") as f:
        saved_dict = pickle.load(f)

    angles = saved_dict['angles']
    nb_range = saved_dict['nb_range']
    borefields = saved_dict['borefields']

    start = time.time()
    size_case_precalc(case_name, case_heat_profile, angles, nb_range, borefields)
    print(time.time() - start)
