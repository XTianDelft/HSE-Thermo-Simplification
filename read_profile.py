import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R_b = 0.1125

df = pd.read_excel("profiles/profile-test.xlsx")
heat_demand = df["Ruimteverwarming_W"]
time = df["time"]

print(f"max hourly demand: {heat_demand.max() / 1e3:.2f} kW")

plt.plot(heat_demand)
plt.show()
