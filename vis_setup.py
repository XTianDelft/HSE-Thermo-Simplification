import numpy as np
import matplotlib.pyplot as plt
import pygfunction as gt

r_b = 0.125 #Borehole radius

# custom field with pygfunction
degangle = 15
tilt = np.deg2rad(degangle)
Nbs = [9] # Number of boreholes to visualize
H = 48.1 # Borehole length (in meters)
Nb = 8   # Number of Boreholes
# print("Borehole Radial Head Separation(Radius): ",B," [m]")


for Nb in Nbs:
    B = 0 if Nb == 1 else (r_b+.05)/np.sin(np.pi/Nb) #Radial Separation of borehole heads

    phis = np.linspace(0, 2*np.pi, Nb, endpoint=False)
    gt_borefield = gt.borefield.Borefield(H, 0, r_b, B*np.cos(phis), B*np.sin(phis), 0 if Nb == 1 else tilt, phis)
    gt_borefield.visualize_field(labels=False)

    fig = plt.gcf()
    ax2 = fig.axes[1]
    ax2.set

    fig.set_size_inches(8, 3.5)
    fp = f'results/setup_vis-tilt={degangle}-Nb={Nb}.pdf'
    print('Saving to', fp)
    plt.savefig(fp)
    plt.show()

# gfunc = gt.gfunction.gFunction(gt_borefield, alpha, method='similarities')
# times = gt.load_aggregation.ClaessonJaved(3600, 1 * 8760 * 3600).get_times_for_simulation()
# gfunc.evaluate_g_function(times)
# gfunc.visualize_g_function()

