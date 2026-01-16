import numpy as np
from heat_profile import *

from bhe_sizing_adv import *

profile_root = 'profiles/'
multi_area_filenames = {
    'tuinzicht': {
        # 'Acaciastraat1': ['A71'],
        # 'Acaciastraat2': ['A69', 'A67'],
        # 'Acaciastraat3': ['A67', 'A67'],
        'Acaciastraat5': ['A67'],
        # 'Acaciastraat6': ['A71', 'A69'],
        # 'Acaciastraat7': ['A67', 'A67', 'A67'],

        # 'Eikstraat1': ['E37', 'E35'],
        # 'Eikstraat2': ['E33', 'E31', 'E35'],
        # 'Eikstraat3': ['E35', 'E25'],
        # 'Eikstraat4': ['E37'],
        'Eikstraat5': ['E35', 'E33'],
        # 'Eikstraat6': ['E31', 'E35', 'E35'],
        # 'Eikstraat7': ['E25'],
    },
    # 'westpolder': {
    #     'Havenstraat1': ['H25', 'H23'],
    #     'Havenstraat2': ['H21', 'H21'],
    #     'Havenstraat3': ['H23', 'H21'],
    #     'Havenstraat5': ['H9'],

    #     'Eilandstraat1': ['E31', 'E29'],
    #     'Eilandstraat3': ['E31', 'E31', 'E29', 'E29'],
    #     'Eilandstraat4': ['E23', 'E21'],
    #     'Eilandstraat6': ['E23', 'E23', 'E21', 'E21'],
    #     'Eilandstraat7': ['E17', 'E17'],
    #     'Eilandstraat8': ['E13'],
    # },
    # 'woudhuis': {
    #     'Kennemerland1': ['K44', 'K46'],
    #     'Kennemerland3': ['K34', 'K32', 'K32', 'K28', 'K28', 'K24'],
    #     'Kennemerland4': ['K34', 'K32', 'K32'],
    #     'Kennemerland5': ['K28', 'K28', 'K24'],
    #     'Kennemerland6': ['K22', 'K20', 'K18', 'K16'],
    #     'Kennemerland7': ['K22', 'K20'],
    #     'Kennemerland8': ['K18', 'K16'],
    # },
    'hillegersberg': {
        'Jubileumplein1': ['J1', 'J2', 'J3', 'J4'],
        # 'Jubileumplein2': ['J4', 'J1', 'J7', 'J7'],
        # 'Jubileumplein3': ['J9', 'J2', 'J2'],
        # 'Jubileumplein4': ['J1', 'J13', 'J14'],
        # 'Jubileumplein5': ['J15', 'J7', 'J1', 'J15'],
        # 'Jubileumplein6': ['J14', 'J2', 'J7', 'J22'],
        # 'Jubileumplein7': ['J15', 'J14', 'J7'],
    }
}

results_root = 'results/'

areas = multi_area_filenames.keys()
print('Areas:', ' '.join(areas))

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(13, 4), sharey=True, gridspec_kw={'width_ratios': [3, 1]})

i = 0
for area in areas:
    cases = multi_area_filenames[area]
    area_root = profile_root + area + '/'

    #size_multiple_cases(cases, area_root, results_root, precalc_fp)
    for case_name, fns in cases.items():
        fns_with_ext = list(map(lambda fn: fn + '.xlsx', fns))
        print(f'Loading and summing profiles for {case_name}:')
        total_hourly_profile = load_profiles_and_sum(fns_with_ext, area_root)/1e3

        zorder = 2.5 - 0.1*i

        plt.sca(ax1)
        plt.plot(total_hourly_profile, lw=.5, zorder=zorder)
        print(case_name, zorder)

        plt.sca(ax2)
        plot_duration_curve(total_hourly_profile, mark_peak=True, label=case_name, zorder=zorder)

        i += 1



ax1.set_xlim(0, 8760)
ax1.set_xlabel('time (hours)')
ax1.set_ylabel('building heat load (kW)')
ax1.grid()

ax2.set_ylim(0, 90)
ax2.legend()

plt.tight_layout()
# plt.subplots_adjust(wspace=.0)
plt.savefig(f'results/duration_plots.pdf')
plt.show()
# plt.clf()



