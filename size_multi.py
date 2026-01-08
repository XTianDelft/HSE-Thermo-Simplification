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
    }
}

# store the house living space, type, and year of construction
multi_house_properties = {
    'tuinzicht': {
        'A71': (60, 'corner', 1939),
        'A69': (60, 'terraced', 1939),
        'A67': (105, 'terraced', 1939),

        'E37': (110, 'corner', 1939),
        'E35': (105, 'terraced', 1939),
        'E33': (115, 'terraced', 1939),
        'E31': (115, 'terraced', 1939),
        'E25': (100, 'corner', 1939),
    }
}

areas = multi_area_filenames.keys()
assert areas == multi_house_properties.keys(), 'filename areas does not match property areas'
print('Areas:', ' '.join(areas))

for area in areas:
    area_profiles_fns = multi_area_filenames[area]
    area_root = profile_root + area + '/'
    for case_name, profile_names in area_profiles_fns.items():
        if case_name != 'Acaciastraat7':
            continue
        fns_with_ext = []
        living_space = 0
        for profile_name in profile_names:
            fns_with_ext.append(profile_name + '.xlsx')
            living_space += multi_house_properties[area][profile_name][0]
        advanced_sizing(fns_with_ext, area_root, case_name, living_space)
