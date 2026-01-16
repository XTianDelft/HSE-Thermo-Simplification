from size_multi import *
from bhe_sizing_vis import *

# visualize all cases in size_multi.py

for area in areas:
    for case in multi_area_filenames[area]:
        results_fp = f'results-{case}.pkl'
        try:
            visualize(results_fp)
        except ValueError as e:
            print(e)
            plt.clf()
