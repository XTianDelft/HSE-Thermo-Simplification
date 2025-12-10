import matplotlib.pyplot as plt
import pandas as pd
import numpy as np





filename = './Trend EM WP.xlsx'
df = pd.read_excel(filename)

print(df.columns.values)

df['Time stamp'] = pd.to_datetime(df['Time stamp'], format='%m/%d/%Y %I:%M:%S %p')
time = df['Time stamp']

d1 = pd.to_numeric(df['01EM2_TTA - Interval Trend Log'], errors='coerce')
d2 = pd.to_numeric(df['01EM2_TTR - Interval Trend Log'], errors='coerce')

plt.plot(time, d1)
plt.plot(time, d2)
plt.show()

exit()
# this file contains the daily heat into or out of the heat pump?
filename = './Productie warmtepomp.xlsx'
df = pd.read_excel(filename)

# print(type(df['Time stamp'][0]))
df['Time stamp'] = pd.to_datetime(df['Time stamp'], format='%m/%d/%Y %I:%M:%S %p')

plt.plot(df['Time stamp'], df['01EM2_Energy Interval Trend Log'])
plt.show()
