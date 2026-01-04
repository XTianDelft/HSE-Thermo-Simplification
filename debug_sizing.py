import pygfunction as gt
import numpy as np
import matplotlib.pyplot as plt


B = .7
phis = np.linspace(0, 2*np.pi, 16, endpoint=False)
borefield = gt.borefield.Borefield(14, 0, .125, B*np.cos(phis), B*np.sin(phis), np.deg2rad(15), phis)


alpha = 1e-6
time_values = gt.load_aggregation.ClaessonJaved(3600, 1 * 8760 * 3600).get_times_for_simulation()
method = 'similarities'

print(time_values)

gfunc = gt.gfunction.gFunction(borefield, alpha, time_values, method=method)
gfunc.visualize_g_function()
plt.show()

'''
  File "/proj/Projects/Uni/EE_BSc/minor/bhe_proj/HSE-Thermo-Simplification/bhe_sizing.py", line 91, in <module>
    length = borefield.size(L4_sizing=True) #each borehole length
  File "/proj/Projects/Uni/EE_BSc/minor/bhe_proj/HSE-Thermo-Simplification/.env/lib/python3.10/site-packages/GHEtool/Borefield.py", line 1108, in size
    length = self.size_L4(H_init, self._calculation_setup.quadrant_sizing)
  File "/proj/Projects/Uni/EE_BSc/minor/bhe_proj/HSE-Thermo-Simplification/.env/lib/python3.10/site-packages/GHEtool/Borefield.py", line 1396, in size_L4
    min_temp, sized = self._size_based_on_temperature_profile(20, hourly=True) if np.any(
  File "/proj/Projects/Uni/EE_BSc/minor/bhe_proj/HSE-Thermo-Simplification/.env/lib/python3.10/site-packages/GHEtool/Borefield.py", line 1479, in _size_based_on_temperature_profile
    self._calculate_temperature_profile(self.H, hourly=True)
  File "/proj/Projects/Uni/EE_BSc/minor/bhe_proj/HSE-Thermo-Simplification/.env/lib/python3.10/site-packages/GHEtool/Borefield.py", line 1812, in _calculate_temperature_profile
    results_old = calculate_temperatures(H, hourly=hourly)
  File "/proj/Projects/Uni/EE_BSc/minor/bhe_proj/HSE-Thermo-Simplification/.env/lib/python3.10/site-packages/GHEtool/Borefield.py", line 1774, in calculate_temperatures
    g_values = self.gfunction(self.load.time_L4, H)
  File "/proj/Projects/Uni/EE_BSc/minor/bhe_proj/HSE-Thermo-Simplification/.env/lib/python3.10/site-packages/GHEtool/Borefield.py", line 1904, in gfunction
    return jit_gfunction_calculation()
  File "/proj/Projects/Uni/EE_BSc/minor/bhe_proj/HSE-Thermo-Simplification/.env/lib/python3.10/site-packages/GHEtool/Borefield.py", line 1886, in jit_gfunction_calculation
    return self.gfunction_calculation_object.calculate(
  File "/proj/Projects/Uni/EE_BSc/minor/bhe_proj/HSE-Thermo-Simplification/.env/lib/python3.10/site-packages/GHEtool/VariableClasses/GFunction.py", line 262, in calculate
    gfunc_uniform_T = gvalues(time_value_new, borefield, alpha, borehole_length, interpolate)
  File "/proj/Projects/Uni/EE_BSc/minor/bhe_proj/HSE-Thermo-Simplification/.env/lib/python3.10/site-packages/GHEtool/VariableClasses/GFunction.py", line 226, in gvalues
    traceback.print_stack()
'''
