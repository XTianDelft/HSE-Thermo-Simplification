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


# returns the baseload per month (in kWh/month) and the peak load per month (in kW/month)
def read_monthly_profile(filename):
    df = pd.read_excel(filename)
    heat_demand = df["Ruimteverwarming_W"]
    time = df["time"]

    hours_per_month = 8760 // 12
    monthly = heat_demand.to_numpy().reshape((12, hours_per_month))

    return monthly.sum(axis=1) / 1e3, monthly.max(axis=1) / 1e3


read_monthly_profile("profiles/profile-test.xlsx")
