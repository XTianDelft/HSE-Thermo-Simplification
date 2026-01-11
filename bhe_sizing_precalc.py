import pickle
from tqdm.contrib.concurrent import process_map
import pygfunction as gt
from GHEtool import *

from heat_profile import *

T_g = 10     # undisturbed ground temperature
Rb = 0.1125  # effective borehole thermal resistance
r_b = 0.125 # Borehole radius

soil_type = 'clay'
match soil_type:
    case 'sand':
        k_g = 2.1771   # ground thermal conductivity
        Cp = 2.9288e6  # ground volumetric heat capacity in J/m3K
    case 'clay':
        k_g = 1.5910   # ground thermal conductivity
        Cp = 2.9288e6  # ground volumetric heat capacity in J/m3K
    case _:
        print(f'unknown soil type {soil_type}')
        exit(1)
ground = GroundConstantTemperature(k_g, T_g, Cp)

# min and max fluid temperatures
Tf_max = 16
Tf_min = 0

max_tilt = 45
max_nb = 20

# number of processes that simultaneously calculate the lengths
# for different tilt angles and number of boreholes
do_parallel = True
nr_of_processes = 6
high_res = True

gfunc_options = {
    'method': 'similarities',
    'linear_threshold': 5*3600
}


if high_res:
    precalc_lengths = np.concat([np.arange(10, 20, 2), np.arange(20, 70, 1), np.arange(70, 100, 2), np.arange(100, 200, 10), np.arange(200, 500+1, 50), [1000, 2000]])
else:
    precalc_lengths = np.concat([np.arange(10, 70, 5), np.arange(70, 100, 10), np.arange(100, 300+1, 50), [500, 1000]])
print(f'{len(precalc_lengths) = }')
def generate_precalc_borefield(inputs):
    degangle, Nb = inputs
    tilt = np.deg2rad(degangle)
    H = 40  # initial height (does not matter)
    B = (r_b+.05)/np.sin(np.pi/Nb)  # radial Separation of borehole heads

    borefield = Borefield()

    borefield.THRESHOLD_WARNING_SHALLOW_FIELD = 10
    borefield.ground_data = ground
    borefield.Rb = Rb
    borefield.set_max_avg_fluid_temperature(Tf_max)
    borefield.set_min_avg_fluid_temperature(Tf_min)

    phis = np.linspace(0, 2*np.pi, Nb, endpoint=False)
    gt_borefield = gt.borefield.Borefield(H, 0, r_b, B*np.cos(phis), B*np.sin(phis), tilt, phis)
    borefield.set_borefield(gt_borefield)

    borefield.create_custom_dataset(borehole_length_array=precalc_lengths, options=gfunc_options)

    gfuncs = borefield.custom_gfunction.gvalues_array
    bh_lens = borefield.custom_gfunction.borehole_length_array
    time_arr = borefield.custom_gfunction.time_array

    if (gfuncs < 0).any():
        print("ERROR: Negative g-functions")
        ids_len = np.where((gfuncs < 0).any(axis=1))[0]
        print(f'negative gfunc for {degangle}°, {Nb} at lengths: {bh_lens[ids_len]}')
        return float('NaN')

    return borefield


# prepare all possible borefield configurations
def precalc_borefields(results_root):
    if high_res:
        angles = np.linspace(0, max_tilt, max_tilt//3+1)
        nb_range = np.arange(1, max_nb+1)
    else:
        angles = np.linspace(0, max_tilt, 10)
        nb_range = np.arange(1, max_nb+1, 2)

    # Generate all the combinations for the inputs as a list of tuples
    ang_vals, nb_vals = np.meshgrid(angles, nb_range, indexing='ij')
    inputs = list(zip(ang_vals.ravel(), nb_vals.ravel()))

    if do_parallel:
        outputs = process_map(generate_precalc_borefield, inputs, max_workers=nr_of_processes)
    else:
        outputs = np.zeros_like(inputs, dtype=np.float64)
        for i, inp in enumerate(inputs):
            print(f'{i+1}/{len(inputs)}')
            outputs[i] = generate_precalc_borefield(inp)

    borefields = np.array(outputs).reshape((len(angles), len(nb_range)))

    saved_dict = {
        'T_g': T_g,
        'k_g': k_g,
        'Cp': Cp,
        'Rb': Rb,
        'r_b': r_b,
        'Tf_min': Tf_min,
        'Tf_max': Tf_max,
        'high_res': high_res,
        'precalc_lengths': precalc_lengths,

        'angles': angles,
        'nb_range': nb_range,
        'borefields': borefields
    }

    results_filename = results_root + f"precalc_borefields-{k_g=:.2f}-{Cp=:.2e}-{T_g=:.1f}-{'HIGHRES' if high_res else 'LOWRES'}.pkl"
    print(f'writing results to {results_filename}')
    with open(results_filename, "wb") as f_out:
        pickle.dump(saved_dict, f_out)


if __name__ == '__main__':
    precalc_borefields('results/')
