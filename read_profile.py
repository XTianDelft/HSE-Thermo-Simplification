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


# returns the peak load per month (in kW) and the total load per month (in MWh)
def read_monthly_profile(filename):
    df = pd.read_excel(filename)
    heat_demand = df["Ruimteverwarming_W"]
    time = df["time"]

    print(df)

    print(f"\nmax hourly demand: {heat_demand.max() / 1e3:.2f} kW")

    hours_per_month = 8760 // 12
    monthly = heat_demand.to_numpy().reshape((12, hours_per_month))

    return monthly.max(axis=1) / 1e3, monthly.sum(axis=1) / 1e6


read_monthly_profile("profiles/profile-test.xlsx")
