from bhe_sizing_adv import *

profile_root = 'profiles/'
multi_area_filenames = {
    'tuinzicht': {
        'Acaciastraat1': ['A71'],
        'Acaciastraat2': ['A69', 'A67'],
        'Acaciastraat3': ['A67', 'A67'],
        'Acaciastraat4': ['A67'],
        'Acaciastraat6': ['A71', 'A69'],
        'Acaciastraat7': ['A67', 'A67', 'A67'],

        'Eikstraat1': ['E37', 'E35'],
        'Eikstraat2': ['E33', 'E31', 'E35'],
        'Eikstraat3': ['E35', 'E25'],
        'Eikstraat4': ['E37'],
        'Eikstraat5': ['E35', 'E33'],
        'Eikstraat6': ['E31', 'E35', 'E35'],
        'Eikstraat7': ['E25'],
    },
    'westpolder': {
        'Havenstraat1': ['H25', 'H23'],
        'Havenstraat2': ['H21', 'H21'],
        'Havenstraat3': ['H23', 'H21'],
        'Havenstraat5': ['H9'],

        'Eilandstraat1': ['E31', 'E29'],
        'Eilandstraat3': ['E31', 'E31', 'E29', 'E29'],
        'Eilandstraat4': ['E23', 'E21'],
        'Eilandstraat6': ['E23', 'E23', 'E21', 'E21'],
        'Eilandstraat7': ['E17', 'E17'],
        'Eilandstraat8': ['E13'],
    }
}

soil_type_lut = {
    'tuinzicht': 'sand',
    'westpolder': 'clay',
}

precalc_fp_lut = {
    'sand': 'precalc_borefields-k_g=2.18-Cp=2.93e+06-T_g=10.0-HIGHRES.pkl',
    'clay': 'precalc_borefields-k_g=1.59-Cp=2.93e+06-T_g=10.0-HIGHRES.pkl'
}

results_root = 'results/'

areas = multi_area_filenames.keys()
print('Areas:', ' '.join(areas))


for area in areas:
    cases = multi_area_filenames[area]
    area_root = profile_root + area + '/'

    soil_type = soil_type_lut[area]
    precalc_fp = results_root + precalc_fp_lut[soil_type]

    size_multiple_cases(cases, area_root, results_root, precalc_fp)
