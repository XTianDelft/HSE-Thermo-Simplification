import matplotlib.pyplot as plt
import numpy as np
import pygfunction as gt

# plot the calculated thermal resistances for the pipes, the COMSOL way as well as the pygfunction way

def COMSOL_convective_thermal_resistances(m_flow_borehole, cross_sect_area, hydraulic_diameter, L_char, mu_f, rho_f, k_f, cp_f):
    Pr = cp_f*mu_f/k_f

    # Q_leg = m_flow_borehole/rho_f
    # v = Q_leg/cross_sect_area
    # Re = rho_f*v*hydraulic_diameter/mu_f

    Re = hydraulic_diameter*m_flow_borehole/(cross_sect_area*mu_f)

    Nu = (3.66 + (0.0668*(hydraulic_diameter/L_char)*Re*Pr)/(1+0.04*((hydraulic_diameter/L_char)*Re * Pr)**(2/3))) if Re < 2300 else (0.023*Re**(0.8)*Pr**(0.4))

    h = Nu*k_f/hydraulic_diameter
    return h

def R_b_pyg(m_flow_borehole, T=3, fully_water=True, do_custom_convection = True):
    k_s = 1.2     # Soil thermal conductivity

    # Borehole dimensions
    r_b = 0.125   # Grout radius
    H = 50        # Length
    D = 0         # Burried depth

    # Pipe dimensions (coaxial)
    r_in_in = 0.022/2    # Inside pipe inner radius
    r_in_out = 0.025/2   # Inside pipe outer radius
    r_mid_in = 0.029/2   # Middle pipe inner radius
    r_mid_out = 0.032/2  # Middle pipe outer radius
    r_out_in = 0.155/2   # Outer pipe inside radius
    r_out_out = 0.160/2  # Outer pipe outside radius

    k_p = 0.19    # Pipe thermal conductivity
    # k_p = 0.05
    k_g = 2.0     # Grout thermal conductivity
    k_gap = 0.026 # Thermal conductivity of the air between the inner and middle pipe

    pos = (0, 0) # position with respect to the borehole
    r_inner = np.array([r_in_in, r_out_in])
    r_outer = np.array([r_mid_out, r_out_out]) # regard the OD of the middle pipe as the OD

    # Fluid properties; NOTE: these are different from the values used in COMSOL
    # https://pygfunction.readthedocs.io/en/stable/modules/media.html
    if fully_water:
        fluid = gt.media.Fluid('MEG', 0, T=T) # water at T=20°C
    else:
        fluid = gt.media.Fluid('MEG', 14, T=T) # 14% ethylene glycol in water at T=20°C
    cp_f = fluid.cp     # Fluid specific isobaric heat capacity
    rho_f = fluid.rho   # Fluid density
    mu_f = fluid.mu     # Fluid dynamic viscosity (kg/m.s)
    k_f = fluid.k       # Fluid thermal conductivity

    # flow_rate = 3/9    # Flow rate in m3/hour; # TODO: verify this
    # m_flow_borehole = flow_rate/3600*rho_f # Total fluid mass flow rate per borehole (kg/s)

    # For the convection and conduction calculations we use the functions from the pygfunction library.
    # These are different from the formulas used in COMSOL.
    epsilon = 1.0e-6  # Pipe roughness (m); TODO: validate this

    # Pipe conduction thermal resistances
    R_in_pipe = gt.pipes.conduction_thermal_resistance_circular_pipe(r_in_in, r_in_out, k_p)
    R_mid_pipe = gt.pipes.conduction_thermal_resistance_circular_pipe(r_mid_in, r_mid_out, k_p)
    R_mid_gap = gt.pipes.conduction_thermal_resistance_circular_pipe(r_in_out, r_mid_in, k_gap)
    R_out_pipe = gt.pipes.conduction_thermal_resistance_circular_pipe(r_out_in, r_out_out, k_p)
    R_grout = gt.pipes.conduction_thermal_resistance_circular_pipe(r_out_out, r_b, k_g)

    # Fluid convection thermal resistances
    if do_custom_convection:
        h_f_in = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
                m_flow_borehole, r_in_in, mu_f, rho_f, k_f, cp_f, epsilon)
        R_in_conv = 1 / (h_f_in * 2 * np.pi * r_in_in)

        h_f_ann_inner, h_f_ann_outer = gt.pipes.convective_heat_transfer_coefficient_concentric_annulus(
                    m_flow_borehole, r_mid_out, r_out_in, mu_f, rho_f, k_f, cp_f, epsilon)
        D_h = 2 * (r_out_in - r_mid_out)

        Nu_a_in = h_f_ann_inner * D_h / k_f
        Nu_a_out = h_f_ann_outer * D_h / k_f
    else:
        Nseg = 10
        L_char = 50/Nseg
        h_f_in = COMSOL_convective_thermal_resistances(m_flow_borehole, np.pi*r_in_in**2, r_in_in*2, L_char, mu_f, rho_f, k_f, cp_f)
        R_in_conv = 1 / (h_f_in * 2 * np.pi * r_in_in)

        # h_f_ann_inner, h_f_ann_outer = gt.pipes.convective_heat_transfer_coefficient_concentric_annulus(
        #             m_flow_borehole, r_mid_out, r_out_in, mu_f, rho_f, k_f, cp_f, epsilon)
        A_ann = np.pi * (r_out_in**2 - r_mid_out**2)
        Dh_ann = 2*(r_out_in - r_mid_out)
        h_f_ann_inner = COMSOL_convective_thermal_resistances(m_flow_borehole, A_ann, Dh_ann, L_char, mu_f, rho_f, k_f, cp_f)
        h_f_ann_outer = COMSOL_convective_thermal_resistances(m_flow_borehole, A_ann, Dh_ann, L_char, mu_f, rho_f, k_f, cp_f)

    R_ann_out_conv = 1 / (h_f_ann_outer * 2 * np.pi * r_out_in)
    R_ann_in_conv = 1 / (h_f_ann_inner * 2 * np.pi * r_mid_out)

    R_ff = R_in_conv + R_in_pipe + R_mid_gap + R_mid_pipe + R_ann_in_conv
    R_fp = R_ann_out_conv + R_out_pipe

    # print(f'{h_f_in = :.3f} {h_f_ann_inner = :.3f} {h_f_ann_outer = :.3f}')
    # print(f'{R_ff = :.3f}, {R_fp = :.3f}, {R_out_pipe = :.3f}, {R_grout = :.3f}, ')

    borehole = gt.boreholes.Borehole(H, D, r_b, 0, 0)
    pipe = gt.pipes.Coaxial(pos, r_inner, r_outer, borehole, k_s, k_g, R_ff, R_fp, J=2)

    R_b = pipe.effective_borehole_thermal_resistance(
        m_flow_borehole, fluid.cp)

    return R_b
R_b_pyg_vec = np.vectorize(R_b_pyg)

def R_b_comsol(m_flow_borehole, T_f=3+273.15, Nseg = 10):
    pi = np.pi
    log = np.log # natural logarithm
    if_ = lambda cond, expr1, expr2: expr1 if cond else expr2

    rho_w = lambda T: 1000*(1 - ((T-273.15+288.9414)/(508929.2*(T-273.15+68.12963)))*(T-273.15-3.9863)**2)
    cp_w = lambda T: 4179.9 - 0.3*(T-273.15) + 0.002*(T-273.15)**2
    k_w = lambda T: 0.561 + 1.9e-3*(T-273.15) - 3.0e-6*(T-273.15)**2
    mu_w = lambda T: 2.414e-5*10**(247.8/((T-273.15)+133.15))

    cp_f = cp_w(T_f)
    k_f = k_w(T_f)
    mu_f = mu_w(T_f)
    rho_f = rho_w(T_f)
    Pr = cp_f*mu_f/k_f

    # COMSOL to python regex:
    # replace: "\t(.*?)(\[.*\])?\t(.*?)\t"
    # with: " = $1\t\t\t# $2 $3 # "
    D_soil = 50			# [m] 50 m # Soil cylinder diameter
    L_soil = 45			# [m] 45 m # Soil thickness
    T_ini = 8			# [degC] 281.15 K # Initial soil temperature
    Tin = 3			# [degC] 276.15 K # Fluid inlet temperature at the borehole top
    # Nbh = 9			#  9  # Number of boreholes in the array
    # Q_sys = 9e-5*9			# [m3/s] 8.1E-4 m³/s # total system volumetric flow delivered to all boreholes
    # Q_bh = Q_sys/Nbh			#  9E-5 m³/s # Volumetric flow per borehole
    # Q_leg = Q_bh			#  9E-5 m³/s # Volumetric flow per leg (down inner / up annulus)
    Q_leg = m_flow_borehole/rho_f
    DI_in = 0.022			# [m] 0.022 m # inner pipe ID (downflow)
    DO_in = 0.025			# [m] 0.025 m # inner pipe OD
    DI_mid = 0.029			# [m] 0.029 m # mid pipe ID
    DO_mid = 0.032			# [m] 0.032 m # mid pipe OD
    DI_out = 0.155			# [m] 0.155 m # outer pipe ID (annulus inner wall)
    DO_out = 0.160			# [m] 0.16 m # outer pipe OD
    k_pipe = 0.19			# [W/(m*K)] 0.19 W/(m·K) # Pipe wall thermal conductivity (PVC default)
    k_mid = 0.19			# [W/(m*K)] 0.19 W/(m·K) # Mid pipe wall thermal conductivity (PVC default)
    k_gap = 0.026			# [W/(m*K)] 0.026 W/(m·K) # Air thermal conductivity
    A_in = pi*DI_in**2/4			#  3.8013E-4 m² # Inner tube cross-sectional area
    v_in = Q_leg/A_in			#  0.23676 m/s # Inner tube mean velocity
    A_ann = pi*(DI_out**2 - DO_mid**2)/4			#  0.018065 m² # Annulus cross-sectional area
    Dh_ann = DI_out - DO_mid			#  0.123 m # Annulus hydraulic diameter
    v_ann = Q_leg/A_ann			#  0.004982 m/s # Annulus mean velocity
    R_in_pipe = log(DO_in/DI_in)/(2*pi*k_pipe)			#  0.10708 s³·K/(kg·m) # Inner pipe wall conduction
    R_mid_gap = if_(DI_mid>DO_in, log(DI_mid/DO_in)/(2*pi*k_gap), 0)			#  0.90853 s³·K/(kg·m) # Conduction across the (optional) small gap between inner pipe OD and mid pipe ID.
    R_mid_pipe = log(DO_mid/DI_mid)/(2*pi*k_mid)			#  0.082459 s³·K/(kg·m) # Conduction through the mid-pipe wall
    R_outer_pipe = log(DO_out/DI_out)/(2*pi*k_pipe)			#  0.026595 s³·K/(kg·m) # Outer pipe wall conduction
    z_top = 0			# [m] 0 m # top surface at z=0; soil goes downward
    x_loc = 0.7			# [m] 0.7 m #
    y_loc = 0.7			# [m] 0.7 m #
    L_Grout = 41			# [m] 41 m # Grout length
    x_loc_1 = .7			# [m] 0.7 m # Loc for new pipes
    y_loc_1 = .7			# [m] 0.7 m # Loc for new pipes
    cp_g = 1800			# [J/(kg*K)] 1800 J/(kg·K) # grout heat capacity
    rho_g = 1000			# [kg/m3] 1000 kg/m³ # grout density
    k_g = 2			# [W/(m*K)] 2 W/(m·K) # grout conductivity
    r_b = 0.125			# [m] 0.125 m # grout radius
    r_oo = DO_out/2			#  0.08 m # outer pipe outer radius.
    t_grout = r_b - r_oo			#  0.045 m # grout thickness (check > 0).
    R_grout = log(r_b/r_oo)/(2*pi*k_g)			#  0.035514 s³·K/(kg·m) # Grout conduction

    # L = 40			# [m] 40 m # BH1 length
    L = 50
    # Nseg = 10			#  10  # BH1: Number of segments
    dz = L/Nseg			#  4 m # BH1: Segment length
    on = 1			#  1  # 1: on/ 0: off
    L_char = dz			#  4 m # set L_char = dz_BHX if you prefer

    Re_in = rho_f*v_in*DI_in/mu_f
    Re_ann = rho_f*v_ann*Dh_ann/mu_f

    Nu_in = if_(Re_in <2300, 3.66 + (0.0668*(DI_in/L_char)*Re_in*Pr)/(1+0.04*((DI_in/L_char)*Re_in * Pr)**(2/3)), 0.023*Re_in**(0.8)*Pr**(0.4))
    Nu_ann = if_(Re_ann <2300, 3.66 + (0.0668*(Dh_ann/L_char)*Re_ann *Pr)/(1+0.04*((Dh_ann/L_char)*Re_ann *Pr)**(2/3)), 0.023*Re_ann **(0.8)*Pr **(0.4))

    h_in = Nu_in*k_f/DI_in
    h_ann = Nu_ann*k_f/Dh_ann

    R_in_conv = 1/(h_in*pi*DI_in)
    R_ann_inner_conv = 1/(h_ann*pi*DO_mid)
    R_ann_outer_conv = 1/(h_ann*pi*DI_out)

    # h_ann_inner_conv, h_ann_outer_conv = gt.pipes.convective_heat_transfer_coefficient_concentric_annulus(
    #             m_flow_borehole, DO_mid/2, DI_out/2, mu_f, rho_f, k_f, cp_f, 1.0e-6)
    # R_ann_inner_conv = 1 / (h_ann_inner_conv * np.pi * DO_mid)
    # R_ann_outer_conv = 1 / (h_ann_outer_conv * np.pi * DI_out)

    R_ann_path = R_ann_outer_conv
    R_inner_path = R_in_conv + R_in_pipe + R_mid_gap + R_mid_pipe + R_ann_inner_conv
    R_b = (1/(1/R_inner_path + 1/R_ann_path)) + R_outer_pipe + R_grout

    return R_b
R_b_comsol_vec = np.vectorize(R_b_comsol)


m_flow_values = np.linspace(0.01, 1, 1000)

plt.plot(m_flow_values, R_b_pyg_vec(m_flow_values, fully_water=True), label='pygfunction (water)')
plt.plot(m_flow_values, R_b_pyg_vec(m_flow_values, fully_water=False), label='pygfunction (14% MEG)')
plt.plot(m_flow_values, R_b_pyg_vec(m_flow_values, fully_water=True, do_custom_convection=False), label='pygfunction (water) with COMSOL convection')


# plt.plot(m_flow_values, pyg_thermal_resistances[0], label='pygfunction 1')
# plt.plot(m_flow_values, pyg_thermal_resistances[1], label='pygfunction 2')


for Nseg in [10]:
    comsol_thermal_resistances = R_b_comsol_vec(m_flow_values, 3+273.15, Nseg)
    plt.plot(m_flow_values, comsol_thermal_resistances, label=f'COMSOL Nseg = {Nseg}')#label=f'COMSOL T = {T} °C')

plt.xlim(0, m_flow_values.max())
plt.ylim(0, .3)
plt.minorticks_on()
plt.grid(which='major', color='gray')
plt.grid(which='minor', color='lightgray')
plt.xlabel('mass flow through one 50 m borehole (kg/s)')
plt.ylabel('effective borehole thermal resistance (m.K/W)')
plt.legend(loc='upper right')
plt.gcf().set_size_inches(6, 4)
plt.tight_layout()
plt.savefig('results/pipe_thermal_resistance.pdf')
# plt.savefig('pipe_thermal_resistance.png', dpi=360)
plt.show()
