# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 08:35:12 2020

@author: ssterl
"""

##########################
#### REVUB initialise ####
##########################

# © 2019 CIREG project
# Author: Sebastian Sterl, Vrije Universiteit Brussel
# This code accompanies the paper "Smart renewable electricity portfolios in West Africa" by Sterl et al.
# All equation, section &c. numbers refer to that paper's Supplementary Information or equivalently the REVUB manual.

import numpy as np
import pandas as pd
import numbers as nb


# %% pre.0) Read Excel files with inputs

# filename of Excel sheets with collected data
filename_parameters = 'parameters_simulation.xlsx'
filename_bathymetry = 'data_bathymetry.xlsx'
filename_CF_solar = 'data_CF_solar.xlsx'
filename_CF_wind = 'data_CF_wind.xlsx'
filename_evaporation = 'data_evaporation.xlsx'
filename_precipitation = 'data_precipitation.xlsx'
filename_inflow = 'data_inflow.xlsx'
filename_outflow_prescribed = 'data_outflow_prescribed.xlsx'
filename_load = 'data_load.xlsx'


# [load] general simulation parameters
parameters_general = pd.read_excel (filename_parameters, sheet_name = 'General parameters', header = None)
parameters_general_list = np.array(parameters_general[0][0:].tolist())
parameters_general_values = np.array(parameters_general[1][0:].tolist())

# [load] static hydropower plant parameters
parameters_hydropower = pd.read_excel (filename_parameters, sheet_name = 'Hydropower plant parameters', header = None)
parameters_hydropower_list = np.array(parameters_hydropower[0][0:].tolist())
parameters_hydropower_values = np.array(parameters_hydropower)[0:,2:]


# [read] parameters on plant name, activation, and cascade situation
HPP_name_temp = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name', True, False)][0].astype(str)
HPP_active = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_active', True, False)][0]
HPP_active_save = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_active', True, False)][0]
HPP_cascade_upstream = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_cascade_upstream', True, False)][0].astype(str)
HPP_cascade_downstream = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_cascade_downstream', True, False)][0].astype(str)

# [ensure] upstream cascade plants do not get activated if the HPPs they feed downstream are deactivated
HPP_cascade_upstream[np.where(HPP_active == 0)] = 'nan'
HPP_cascade_upstream = np.delete(HPP_cascade_upstream, np.where(HPP_cascade_upstream == 'nan')[0])

# [ensure] downstream cascade plants do not get activated if the HPPs they are fed from upstream are deactivated
HPP_cascade_downstream[np.where(HPP_active == 0)] = 'nan'
HPP_cascade_downstream = np.delete(HPP_cascade_downstream, np.where(HPP_cascade_downstream == 'nan')[0])

# [provide] specific identifier to plants in a cascade, if the HPPs they feed downstream are activated
for n in range(len(HPP_cascade_upstream)): HPP_active[np.where(HPP_name_temp == HPP_cascade_upstream[n])[0]] = -1
for m in range(len(HPP_cascade_downstream)): HPP_active[np.where(HPP_name_temp == HPP_cascade_downstream[m])[0]] = -2

# [delete] deactivated deselected plants
parameters_hydropower_values = np.delete(parameters_hydropower_values, np.where(HPP_active == 0)[0],1)
HPP_active_save = np.delete(HPP_active_save, np.where(HPP_active == 0)[0])
HPP_active = np.delete(HPP_active, np.where(HPP_active == 0)[0])

# [swap] positions so the plants with upstream cascade function come last in the simulation
parameters_hydropower_values = np.concatenate((np.delete(parameters_hydropower_values, np.where(HPP_active != 1)[0],1), np.delete(parameters_hydropower_values, np.where(HPP_active != -1)[0],1), np.delete(parameters_hydropower_values, np.where(HPP_active != -2)[0],1)), axis = 1)
HPP_active_save = np.concatenate((np.delete(HPP_active_save, np.where(HPP_active != 1)[0],0), np.delete(HPP_active_save, np.where(HPP_active != -1)[0],0), np.delete(HPP_active_save, np.where(HPP_active != -2)[0],0)), axis = 0)
HPP_active = np.concatenate((np.delete(HPP_active, np.where(HPP_active != 1)[0],0), np.delete(HPP_active, np.where(HPP_active != -1)[0],0), np.delete(HPP_active, np.where(HPP_active != -2)[0],0)), axis = 0)


# %% pre.1) Time-related parameters

# [set by user] number of hydropower plants in this simulation
HPP_number = len(parameters_hydropower_values[0,:])
HPP_number_run = np.sum(HPP_active_save)

# [set by user] the reference years used in the simulation
year_start = int(parameters_general_values[np.where(parameters_general_list == 'year_start', True, False)][0])
year_end = int(parameters_general_values[np.where(parameters_general_list == 'year_end', True, False)][0])
simulation_years = list(range(year_start, year_end + 1))

column_start = int(parameters_general_values[np.where(parameters_general_list == 'column_start', True, False)][0])
column_end = column_start + len(simulation_years) - 1

# [constant] number of hours in a day
hrs_day = 24

# [constant] number of months in a year
months_yr = 12

# [constant] number of seconds and minutes in an hour
secs_hr = 3600
mins_hr = 60

# [preallocate] number of days in each year
days_year = np.zeros(shape = (months_yr, len(simulation_years)))
hrs_byyear = np.zeros(shape = len(simulation_years))

# [calculate] for each year in the simulation: determine if leap year or not;
# write corresponding amount of hours into hrs_byyear
for y in range(len(simulation_years)):
    if np.ceil(simulation_years[y]/4) == simulation_years[y]/4 and np.ceil(simulation_years[y]/100) != simulation_years[y]/100:
        days_year[:,y] = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        days_year[:,y] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hrs_byyear[y] = sum(days_year[:,y])*hrs_day

# [arrange] for data arrangements in matrices: determine hours corresponding to start of each month
# (e.g. January = 0; February = 744; March = 1416 or 1440 depending on whether leap year or not; &c.)
positions = np.zeros(shape = (len(days_year) + 1, len(simulation_years)))
positions[0,:] = 0
for y in range(len(simulation_years)):
    for n in range(len(days_year)):
        positions[n+1,y] = hrs_day*days_year[n,y] + positions[n,y]


# %% pre.2) Model parameters
        
##### GENERAL HYDROPOWER DATA #####

# [set by user] wish to run in calibration mode only (i.e. CONV scenario only)? (0 = no, 1 = yes)
calibration_only = int(parameters_general_values[np.where(parameters_general_list == 'calibration_only', True, False)][0])

# [set by user] wish to model pumped storage (Note 7) or not? (0 = no, 1 = yes)
option_storage = int(parameters_general_values[np.where(parameters_general_list == 'option_storage', True, False)][0])

# [constant] density of water (kg/m^3) (introduced in eq. S3)
rho = parameters_general_values[np.where(parameters_general_list == 'rho', True, False)][0]

# [constant] gravitational acceleration (m/s^2) (introduced in eq. S8)
g = parameters_general_values[np.where(parameters_general_list == 'g', True, False)][0]

# [set by user] percentile value used to calculate exceedance probability of delivered power
p_exceedance = parameters_general_values[np.where(parameters_general_list == 'p_exceedance', True, False)][0]

##### HYDROPOWER OPERATION PARAMETERS #####

# [set by user] threshold for determining whether HPP is "large" or "small" - if
# t_fill (eq. S1) is larger than threshold, classify as "large"
T_fill_thres = parameters_general_values[np.where(parameters_general_list == 'T_fill_thres', True, False)][0]

# [set by user] optional: requirement on Loss of Energy Expectation (criterion (ii) in Figure S1).
# As default, the HSW mix does not allow for any LOEE. However, this criterion could be relaxed.
# E.g. LOEE_allowed = 0.01 would mean that criterion (ii) is relaxed to 1% of yearly allowed LOEE instead of 0%.
LOEE_allowed = parameters_general_values[np.where(parameters_general_list == 'LOEE_allowed', True, False)][0]

# [set by user] prevent hydropower blackouts to be potentially worse under BAL/STOR than under CONV if turned on
prevent_droughts_increase = parameters_general_values[np.where(parameters_general_list == 'prevent_droughts_increase', True, False)][0]

##### SIMULATION ACCURACY PARAMETERS #####

# [set by user] This number defines the amount of discrete steps between 0 and max(E_hydro + E_solar + E_wind)
# reflecting the accuracy of determining the achieved ELCC
N_ELCC = int(parameters_general_values[np.where(parameters_general_list == 'N_ELCC', True, False)][0])

# [set by user] Number of loops for iterative estimation of P_stable,BAL/STOR (see eq. S9 & explanation below eq. S19)
# Typically, 2-3 iterations suffice until convergence is achieved.
X_max = int(parameters_general_values[np.where(parameters_general_list == 'X_max', True, False)][0])

# [set by user] When min(Psi) (eq. S21) is lower than this threshold, no further refinement loops
# are performed. This number can be increased to speed up the simulation.
psi_min_threshold = parameters_general_values[np.where(parameters_general_list == 'psi_min_threshold', True, False)][0]


# %% pre.3) Static parameters

# [set by user] name of hydropower plant
HPP_name = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name', True, False)][0].astype(str)
HPP_cascade_upstream = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_cascade_upstream', True, False)][0].astype(str)
HPP_cascade_downstream = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_cascade_downstream', True, False)][0].astype(str)
HPP_name_data_inflow = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name_data_inflow', True, False)][0].tolist()
HPP_name_data_precipitation = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name_data_precipitation', True, False)][0].tolist()
HPP_name_data_evaporation = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name_data_evaporation', True, False)][0].tolist()
HPP_name_data_outflow_prescribed = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name_data_outflow_prescribed', True, False)][0].tolist()
HPP_name_data_CF_solar = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name_data_CF_solar', True, False)][0].tolist()
HPP_name_data_CF_wind = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name_data_CF_wind', True, False)][0].tolist()
HPP_name_data_load = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name_data_load', True, False)][0].tolist()
HPP_name_data_bathymetry = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name_data_bathymetry', True, False)][0].tolist()

# [set by user] relative capacity of solar vs. wind to be installed (cf. explanation below eq. S13)
# e.g. c_solar_relative = 1 means only solar deployment, no wind deployment on the grid of each HPP in question.
c_solar_relative = parameters_hydropower_values[np.where(parameters_hydropower_list == 'c_solar_relative', True, False)][0]
c_wind_relative = 1 - c_solar_relative

# [set by user] the parameter f_reg controls the fraction (<=1) of average inflow allocated to flexible use. Code B will enter a default if left empty by user
f_reg = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_reg', True, False)][0]

# [set by user] maximum head (m)
h_max = parameters_hydropower_values[np.where(parameters_hydropower_list == 'h_max', True, False)][0]

# [set by user] maximum lake area (m^2)
A_max = parameters_hydropower_values[np.where(parameters_hydropower_list == 'A_max', True, False)][0]
A_max_cumul = parameters_hydropower_values[np.where(parameters_hydropower_list == 'A_max', True, False)][0]

# [set by user] maximum storage volume (m^3) and initial filling fraction
V_max = parameters_hydropower_values[np.where(parameters_hydropower_list == 'V_max', True, False)][0]
V_max_cumul = parameters_hydropower_values[np.where(parameters_hydropower_list == 'V_max', True, False)][0]
f_initial_frac = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_initial_frac', True, False)][0]

# [set by user] turbine capacity (MW)
P_r_turb = parameters_hydropower_values[np.where(parameters_hydropower_list == 'P_r_turb', True, False)][0]

# [set by user] if using STOR scenario: lower reservoir volume (m^3) and initial filling fraction
V_lower_max = parameters_hydropower_values[np.where(parameters_hydropower_list == 'V_lower_max', True, False)][0]
V_lower_initial_frac = parameters_hydropower_values[np.where(parameters_hydropower_list == 'V_lower_initial_frac', True, False)][0]

# [set by user] if using STOR scenario: pump capacity (MW)
P_r_pump = parameters_hydropower_values[np.where(parameters_hydropower_list == 'P_r_pump', True, False)][0]

# [set by user] turbine and pump throughput (m^3/s, see explanation following eq. S8)
Q_max_turb = parameters_hydropower_values[np.where(parameters_hydropower_list == 'Q_max_turb', True, False)][0]
Q_max_pump = parameters_hydropower_values[np.where(parameters_hydropower_list == 'Q_max_pump', True, False)][0]

# [set by user] turbine efficiency (introduced in eq. S8)
eta_turb = parameters_hydropower_values[np.where(parameters_hydropower_list == 'eta_turb', True, False)][0]

# [set by user] pumping efficiency
eta_pump = parameters_hydropower_values[np.where(parameters_hydropower_list == 'eta_pump', True, False)][0]

# [set by user] ramp rate restrictions (eq. S16, S37): fraction of full capacity per minute
dP_ramp_turb = parameters_hydropower_values[np.where(parameters_hydropower_list == 'dP_ramp_turb', True, False)][0]
dP_ramp_pump = parameters_hydropower_values[np.where(parameters_hydropower_list == 'dP_ramp_pump', True, False)][0]

# [set by user] minimum load of individual hydropower turbines (lower bound of operating range)
min_load_turbine = parameters_hydropower_values[np.where(parameters_hydropower_list == 'min_load_turbine', True, False)][0]

# [set by user] minimum required environmental outflow fraction (eq. S4, S5)
d_min = parameters_hydropower_values[np.where(parameters_hydropower_list == 'd_min', True, False)][0]

# [set by user] alpha (eq. S6) for conventional HPP operation rule curve (eq. S4)
alpha = parameters_hydropower_values[np.where(parameters_hydropower_list == 'alpha', True, False)][0]

# [set by user] gamma (eq. S4) for conventional HPP operation rule curve (eq. S4)
gamma_hydro = parameters_hydropower_values[np.where(parameters_hydropower_list == 'gamma_hydro', True, False)][0]

# [set by user] optimal filling fraction f_opt (eq. S4, S5)
f_opt = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_opt', True, False)][0]

# [set by user] fraction f_spill beyond which spilling starts (eq. S7)
f_spill = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_spill', True, False)][0]

# [set by user] mu parameter to control spilling (eq. S7)
mu = parameters_hydropower_values[np.where(parameters_hydropower_list == 'mu', True, False)][0]

# [set by user] thresholds f_stop and f_restart (see page 4) for stopping and restarting
# hydropower production to maintain minimum drawdown levels
f_stop = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_stop', True, False)][0]
f_stop_cumul = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_stop', True, False)][0]
f_restart = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_restart', True, False)][0]
f_restart_cumul = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_restart', True, False)][0]

# [set by user] the parameter f_size controls allowed VRE overproduction and is the percentile value described in eq. S11
f_size = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_size', True, False)][0]

# [set by user] number of turbines per hydropower plant (used for turbine use statistics in script C)
no_turbines = parameters_hydropower_values[np.where(parameters_hydropower_list == 'no_turbines', True, False)][0]

# [set by user] percentile value used to calculate exceedance probability of delivered power
year_calibration_start = parameters_hydropower_values[np.where(parameters_hydropower_list == 'year_calibration_start', True, False)][0]

# [set by user] percentile value used to calculate exceedance probability of delivered power
year_calibration_end = parameters_hydropower_values[np.where(parameters_hydropower_list == 'year_calibration_end', True, False)][0]


# %% pre.4) Time series

# [preallocate]
Q_in_nat_hourly = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))
precipitation_flux_hourly = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))
evaporation_flux_hourly = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))
Q_out_stable_env_irr_hourly = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))
CF_solar_hourly = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))
CF_wind_hourly = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))
L_norm = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))

# [preallocate] corrector in case no-VRE scenario
c_VRE_corrector = np.full([HPP_number], np.nan)

# [loop] over all HPP names to extract relevant time series from user-prepared Excel tables
for n in range(len(HPP_name)):
    
    # [set by user] natural inflow and prescribed environmental/irrigation outflow at hourly timescale (m^3/s)
    if str(HPP_name_data_inflow[n]) != 'nan':
        Q_in_nat_hourly[:,:,n] = pd.read_excel (filename_inflow, sheet_name = HPP_name_data_inflow[n], header = None, usecols = range(column_start - 1, column_end), nrows = int(np.max(positions)))
    
    # [set by user] Precipitation flux (kg/m^2/s)
    if str(HPP_name_data_precipitation[n]) != 'nan':
        precipitation_flux_hourly[:,:,n] = pd.read_excel (filename_precipitation, sheet_name = HPP_name_data_precipitation[n], header = None, usecols = range(column_start - 1, column_end), nrows = int(np.max(positions)))
        
    # [set by user] Evaporation flux (kg/m^2/s)
    if str(HPP_name_data_evaporation[n]) != 'nan':
        evaporation_flux_hourly[:,:,n] = pd.read_excel (filename_evaporation, sheet_name = HPP_name_data_evaporation[n], header = None, usecols = range(column_start - 1, column_end), nrows = int(np.max(positions)))

    if str(HPP_name_data_outflow_prescribed[n]) != 'nan':
        Q_out_stable_env_irr_hourly[:,:,n] = pd.read_excel (filename_outflow_prescribed, sheet_name = HPP_name_data_outflow_prescribed[n], header = None, usecols = range(column_start - 1, column_end), nrows = int(np.max(positions)))
    
    # [set by user] capacity factors for solar and wind power (eq. S12)
    if str(HPP_name_data_CF_solar[n]) != 'nan' and str(HPP_name_data_CF_wind[n]) != 'nan':
        
        CF_solar_hourly[:,:,n] = pd.read_excel (filename_CF_solar, sheet_name = HPP_name_data_CF_solar[n], header = None, usecols = range(column_start - 1, column_end), nrows = int(np.max(positions)))
        CF_wind_hourly[:,:,n] = pd.read_excel (filename_CF_wind, sheet_name = HPP_name_data_CF_wind[n], header = None, usecols = range(column_start - 1, column_end), nrows = int(np.max(positions)))
        c_VRE_corrector[n] = 1
        
    # [set by user] in case solar field left empty, simulate only with wind
    elif str(HPP_name_data_CF_solar[n]) == 'nan' and str(HPP_name_data_CF_wind[n]) != 'nan':  
        
        CF_solar_hourly[:,:,n] = np.ones(shape = (int(np.max(positions)), len(simulation_years)))
        CF_wind_hourly[:,:,n] = pd.read_excel (filename_CF_wind, sheet_name = HPP_name_data_CF_wind[n], header = None, usecols = range(column_start - 1, column_end), nrows = int(np.max(positions)))
        c_VRE_corrector[n] = 1
        
        c_solar_relative[n] = 0
        c_wind_relative[n] = 1 - c_solar_relative[n]
    
    # [set by user] in case wind field left empty, simulate only with solar
    elif str(HPP_name_data_CF_solar[n]) != 'nan' and str(HPP_name_data_CF_wind[n]) == 'nan':  
        
        CF_solar_hourly[:,:,n] = pd.read_excel (filename_CF_solar, sheet_name = HPP_name_data_CF_solar[n], header = None, usecols = range(column_start - 1, column_end), nrows = int(np.max(positions)))
        CF_wind_hourly[:,:,n] = np.ones(shape = (int(np.max(positions)), len(simulation_years)))
        c_VRE_corrector[n] = 1
        
        c_solar_relative[n] = 1
        c_wind_relative[n] = 1 - c_solar_relative[n]
    
    # [allow] simulating hydro-only flexibility scenarios without solar and wind by setting dummy constant time series for VRE
    elif str(HPP_name_data_CF_solar[n]) == 'nan' and str(HPP_name_data_CF_wind[n]) == 'nan':
        
        # [set] dummy values for VRE
        CF_solar_hourly[:,:,n] = np.ones(shape = (int(np.max(positions)), len(simulation_years)))
        CF_wind_hourly[:,:,n] = np.ones(shape = (int(np.max(positions)), len(simulation_years)))
        c_VRE_corrector[n] = 0
        
        # [set] dummy c_solar and c_wind
        c_solar_relative[n] = 0
        c_wind_relative[n] = 0
    
    # [set by user] Load curves (L_norm; see eq. S10)
    if str(HPP_name_data_load[n]) != 'nan':
        L_norm[:,:,n] = pd.read_excel (filename_load, sheet_name = HPP_name_data_load[n], header = None, usecols = range(column_start - 1, column_end), nrows = int(np.max(positions)))


# %% pre.5) Simulation accuracy

# [set by user] These values are used to get a good initial guess for the order of magnitude of the ELCC.
# This is done by multiplying them with yearly average E_{hydro}.
# A suitable range within which to identify the optimal solution (eq. S21) is thus obtained automatically
# for each HPP, regardless of differences in volume, head, rated power, &c.
# The value f_init_BAL_end may have to be increased in scenarios where the ELCC becomes extremely high,
# e.g. when extremely good balancing sources other than hydro are present.
# For the scenarios in (Sterl et al.), the ranges 0-3 work for all HPPs. Usually, smaller ranges can be chosen.
f_init_BAL_start = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_init_BAL_start', True, False)][0]
f_init_BAL_step = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_init_BAL_step', True, False)][0]
f_init_BAL_end = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_init_BAL_end', True, False)][0]

# Idem for the optional STOR scenario
f_init_STOR_start = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_init_STOR_start', True, False)][0]
f_init_STOR_step = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_init_STOR_step', True, False)][0]
f_init_STOR_end = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_init_STOR_end', True, False)][0]

# [set by user] Number of refinement loops for equilibrium search for min(Psi) (see eq. S21)
# Every +1 increases precision by one digit. Typically, 2 or 3 iterations suffice. 
N_refine_BAL = parameters_hydropower_values[np.where(parameters_hydropower_list == 'N_refine_BAL', True, False)][0]
N_refine_STOR = parameters_hydropower_values[np.where(parameters_hydropower_list == 'N_refine_STOR', True, False)][0]


# %% pre.6) Bathymetry

# [preallocate]
temp_length_array = np.zeros(HPP_number)

# [loop] over all HPP bathymetries to find maximum length of necessary arrays
for n in range(len(HPP_name)):
    
    # [set by user] head-area-volume curves used during simulations
    if str(HPP_name_data_load[n]) != 'nan':
        temp = pd.read_excel (filename_bathymetry, sheet_name = HPP_name_data_bathymetry[n], header = None)
        temp_length_array[n] = len(temp)
    else:
        # [set] exception in case of RoR plant without reservoir
        temp_length_array[n] = 1
    

# [preallocate]
calibrate_volume = np.full([int(np.max(temp_length_array)), HPP_number], np.nan)
calibrate_area = np.full([int(np.max(temp_length_array)), HPP_number], np.nan)
calibrate_head = np.full([int(np.max(temp_length_array)), HPP_number], np.nan)

# [loop] over all HPP names to extract relevant head-area-volume curves from user-prepared Excel files
for n in range(len(HPP_name)):
    
    # [set by user] head-area-volume curves used during simulations
    
    if str(HPP_name_data_load[n]) != 'nan':
        temp = pd.read_excel (filename_bathymetry, sheet_name = HPP_name_data_bathymetry[n], header = None)
        
        # [extract] volume (m^3)
        calibrate_volume[0:len(temp.iloc[:,0]),n] = temp.iloc[:,0]
        if np.nanmax(calibrate_volume[:,n]) != V_max[n]:
            print('> Warning: V_max inconsistent with maximum value in bathymetry for', HPP_name[n])
        
        # [extract] area (m^2)
        calibrate_area[0:len(temp.iloc[:,1]),n] = temp.iloc[:,1]
        if np.nanmax(calibrate_area[:,n]) != A_max[n]:
            print('> Warning: A_max inconsistent with maximum value in bathymetry for', HPP_name[n])
        
        # [extract] head (m)
        calibrate_head[0:len(temp.iloc[:,2]),n] = temp.iloc[:,2]
        if np.nanmax(calibrate_head[:,n]) != h_max[n]:
            print('> Warning: h_max inconsistent with maximum value in bathymetry for', HPP_name[n])
    
    else:
        
        # [set] exception in case of RoR plant without reservoir
        V_max[n] = 0
        A_max[n] = 0
        f_reg[n] = 0
        calibrate_volume[0,n] = V_max[n]
        calibrate_area[0,n] = A_max[n]
        calibrate_head[0,n] = h_max[n]

