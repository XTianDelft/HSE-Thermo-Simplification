import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

months = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]

def read_hourly_profile(filename):
    df = pd.read_excel(filename)
    heat_demand = df["Ruimteverwarming_W"].to_numpy()
    # time = df["time"]
    return heat_demand

def plot_duration_curve(heat_demand, mark_peak=False, plot_zeros=True, **kwargs):
    duration_curve = np.sort(heat_demand)
    first_0 = duration_curve.searchsorted(0, side='right')
    if not plot_zeros:
        duration_curve = duration_curve[first_0-1:]
    duration_curve = duration_curve[::-1]
    plt.plot(duration_curve, **kwargs)
    if mark_peak:
        plt.scatter(0, (duration_curve)[0], zorder=400, clip_on=False)
    plt.xlim(0, len(duration_curve))
    plt.xlabel('hours of the year')
    plt.ylim(0, None)
    # plt.ylabel('heating load')
    plt.grid()
    plt.tight_layout()

# returns the baseload per month (in kWh/month) and the peak load per month (in kW/month)
def read_monthly_profile(filename):
    hourly_profile = read_hourly_profile(filename)
    hours_per_month = 8760 // 12
    monthly = hourly_profile.reshape((12, hours_per_month))

    return monthly.sum(axis=1) / 1e3, monthly.max(axis=1) / 1e3

def hourly_to_monthly(hourly_profile):
    hours_per_month = 8760 // 12
    monthly = hourly_profile.reshape((12, hours_per_month))
    return monthly.sum(axis=1) / 1e3, monthly.max(axis=1) / 1e3
