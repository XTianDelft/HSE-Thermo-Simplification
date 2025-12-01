import numpy as np
import pygfunction as gt
from GHEtool import *

from heat_profile import *

# min and max fluid temperatures
Tf_max = 16
Tf_min = 0

profile_fn = "profiles/profile-test.xlsx"

hourly_profile = read_hourly_profile(profile_fn)
plot_duration_curve(hourly_profile)
# plot_duration_curve(read_hourly_profile("profiles/profile-test-capped.xlsx"))

load_heat, peak_heat = read_monthly_profile(profile_fn)
load = MonthlyGeothermalLoadAbsolute(load_heat, None, peak_heat, None)
print(f'peak load: {peak_heat.max() :.2f} kW')

# plt.plot(np.arange(12)*8760/12, np.sort(load_heat)[::-1]/(8760/12))
plt.show()
