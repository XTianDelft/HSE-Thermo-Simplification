from size_multi import *
from bhe_sizing_vis import *

for area in areas:
    for case in multi_area_filenames[area]:
        results_fp = f'results-{case}.pkl'
        try:
            visualize(results_fp)
        except ValueError:
            plt.clf()
