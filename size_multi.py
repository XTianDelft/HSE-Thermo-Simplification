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
    },
    'woudhuis': {
        'Kennemerland1': ['K44', 'K46'],
        'Kennemerland3': ['K34', 'K32', 'K32', 'K28', 'K28', 'K24'],
        'Kennemerland4': ['K34', 'K32', 'K32'],
        'Kennemerland5': ['K28', 'K28', 'K24'],
        'Kennemerland6': ['K22', 'K20', 'K18', 'K16'],
        'Kennemerland7': ['K22', 'K20'],
        'Kennemerland8': ['K18', 'K16'],
    },
    'hillegersberg': {
        'Jubileumplein1': ['J1', 'J2', 'J3', 'J4'],
        'Jubileumplein2': ['J4', 'J1', 'J7', 'J7'],
        'Jubileumplein3': ['J9', 'J2', 'J2'],
        'Jubileumplein4': ['J1', 'J13', 'J14'],
        'Jubileumplein5': ['J15', 'J7', 'J1', 'J15'],
        'Jubileumplein6': ['J14', 'J2', 'J7', 'J22'],
        'Jubileumplein7': ['J15', 'J14', 'J7'],
    }
}

soil_type_lut = {
    'tuinzicht': 'sand',
    'westpolder': 'clay',
    'woudhuis': 'sand',
    'hillegersberg': 'sand',
}

precalc_fp_lut = {
    'sand': 'precalc_borefields-k_g=2.18-Cp=2.93e+06-T_g=10.0-HIGHRES.pkl',
    'clay': 'precalc_borefields-k_g=1.59-Cp=2.93e+06-T_g=10.0-HIGHRES.pkl'
}

results_root = 'results/'

areas = multi_area_filenames.keys()

if __name__ == '__main__':
    print('Areas:', ' '.join(areas))


    for area in areas:
        cases = multi_area_filenames[area]
        area_root = profile_root + area + '/'

        soil_type = soil_type_lut[area]
        precalc_fp = results_root + precalc_fp_lut[soil_type]

        size_multiple_cases(cases, area_root, results_root, precalc_fp)
