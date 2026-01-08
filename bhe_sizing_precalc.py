import pickle
from tqdm.contrib.concurrent import process_map
import pygfunction as gt
from GHEtool import *

from heat_profile import *

T_g = 10     # undisturbed ground temperature
k_g = 2.8   # ground thermal conductivity
Cp = 2.8e6  # ground volumetric heat capacity in J/m3K
Rb = 0.1125
r_b = 0.125 # Borehole radius

# min and max fluid temperatures
Tf_max = 16
Tf_min = 0

max_tilt = 45
max_nb = 20

# number of processes that simultaneously calculate the lengths
# for different tilt angles and number of boreholes
do_parallel = True
nr_of_processes = 8
high_res = False

gfunc_options = {'linear_threshold': 10*3600}

ground = GroundConstantTemperature(k_g, T_g, Cp)

if high_res:
    precalc_lengths = np.concat([np.arange(5, 20, 2), np.arange(20, 70, .5), np.arange(70, 100, 2), np.arange(100, 300+1, 10)])
else:
    precalc_lengths = np.concat([np.arange(5, 20, 5), np.arange(20, 70, 2), np.arange(70, 10, 10)], np.arange(100, 300+1, 50))
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
    borefield.set_options_gfunction_calculation(options=gfunc_options)

    borefield.create_custom_dataset(borehole_length_array=precalc_lengths, options={'method': 'similarities'})

    return borefield


# prepare all possible borefield configurations
def precalc_borefields(results_root):
    if high_res:
        angles = np.linspace(0, max_tilt, max_tilt+1)
        nb_range = np.arange(1, max_nb+1)
    else:
        angles = np.linspace(0, max_tilt, 10)
        nb_range = np.arange(1, max_nb+1, 1)

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
