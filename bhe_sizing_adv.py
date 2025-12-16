import pandas as pd
import pygfunction as gt
from GHEtool import *

from heat_profile import *

#--------------------Initial Conditions----------------------------
T_g = 10     # undisturbed ground temperature
k_g = 2.8   # ground thermal conductivity
Cp = 2.8e6  # ground volumetric heat capacity in J/m3K
Rb = 0.1125
r_b = 0.125 #Borehole radius
# min and max fluid temperatures
Tf_max = 16
Tf_min = 0
#===Property Boundaries===
R_max = 30 #Max Radius in m
H_max = 40 #Max depth in m

#==Done Setting, then Importing==
borefield = Borefield(load=HourlyBuildingLoad(read_hourly_profile("profiles/profile-test.xlsx")/1e3))
borefield.ground_data = GroundConstantTemperature(k_g, T_g, Cp)
borefield.Rb = Rb
borefield.set_max_avg_fluid_temperature(Tf_max)
borefield.set_min_avg_fluid_temperature(Tf_min)
#--------------------Initial Conditions----------------------------END
def fitment(angle,length,max_radius,max_depth):#check if the borehole fits in the preset cylindrcal volume
    fit = True
    radangle = np.deg2rad(angle)
    if np.sin(radangle)*length+r_b > max_radius:#radius check
        fit = False
        print("Too far horizontally!")
    elif np.cos(radangle)*length > max_depth:#depth check
        fit = False
        print("Too far vertically!")
    else:
        print("It fits!")
    return fit

def calculate(degangle, Nb): # custom field with pygfunction

    tilt = np.deg2rad(degangle)
    H = 30 #Arbitrary Initial Borehole length (in meters) !!AUTO!!
    B = r_b/np.sin(np.pi/Nb) #Radial Separation of borehole heads !!AUTO!!
    #print("Borehole Radial Head Separation(Radius): ",B," [m]")

    boreholes = [gt.boreholes.Borehole(H, 0, r_b, B*np.cos(phi), B*np.sin(phi), tilt=tilt, orientation=phi) for phi in np.linspace(0, 2*np.pi, Nb, endpoint=False)]
    borefield.set_borefield(gt.borefield.Borefield.from_boreholes(boreholes))

    length = borefield.size(L4_sizing=True) #each borehole length
    #print(f"Each BH is: {length} m")
    length_total = length*Nb

    return length_total,fitment(degangle,length,R_max,H_max)


#-----------------variable Conditions-----------------ONLY FOR DEBUGGING

#for Nb in range(0,17):
    #for degangle in range(0,46,5):
        #print(Nb,degangle)

#Nb = 9 #Number of Boreholes
#degangle = 45 #degrees from vertical
#print ("Results[Length total BH, Fitment]: ", calculate(degangle, Nb))

# ---------------------------------!!!AI Code for Excel Output BELOW!!!---------------------------------------

# 1. Define your axes
angles = list(range(0, 50, 5))  # 0, 5, 10 ... 45
nb_range = list(range(1, 17))  # 1, 2, 3 ... 16

# 2. Create the Excel file and setup formatting
file_name = "Fitment_Results.xlsx"
writer = pd.ExcelWriter(file_name, engine='xlsxwriter')

# Create a dummy dataframe to initialize the sheet structure
df = pd.DataFrame(index=angles, columns=nb_range)
df.to_excel(writer, sheet_name='Data')

workbook = writer.book
worksheet = writer.sheets['Data']

# Define the colors
green_fmt = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100', 'border': 1})
red_fmt = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'border': 1})

# 3. Label the Axes
worksheet.write(0, 0, 'Angle \ Nb')

# 4. Loop through the grid and fill cells
for r_idx, angle in enumerate(angles):
    for c_idx, nb in enumerate(nb_range):
        # Use your functions
        val, is_fit = calculate(angle,nb)

        # Choose format based on fitment
        cell_fmt = green_fmt if is_fit else red_fmt

        # Write to Excel (r_idx+1/c_idx+1 to skip the headers)
        worksheet.write(r_idx + 1, c_idx + 1, val, cell_fmt)

        print("Tilt(Deg): ",angle,"// BH#: ",nb)
        print("Results: ", val, " [m] total, Fitment:", is_fit)

# 5. Save and Close
writer.close()
print(f"Success! Excel sheet '{file_name}' has been created.")