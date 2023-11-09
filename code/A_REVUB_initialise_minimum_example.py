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

# [remove] deactivated hydropower plants
HPP_active = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_active', True, False)][0]
parameters_hydropower_values = np.delete(parameters_hydropower_values, np.where(HPP_active == 0)[0],1)

# [load] simulation accuracy parameters (used in script B)
parameters_simulation = pd.read_excel (filename_parameters, sheet_name = 'Simulation accuracy', header = None)
parameters_simulation_list = np.array(parameters_simulation[0][0:].tolist())
parameters_simulation_values = np.array(parameters_simulation[1][0:].tolist())


# %% pre.1) Time-related parameters

# [set by user] number of hydropower plants in this simulation
HPP_number = len(parameters_hydropower_values[0,:])

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
    if np.ceil(simulation_years[y]/4) == simulation_years[y]/4 and np.ceil(simulation_years[y]/100) != simulation_years[y]/4:
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
        
# [set by user] wish to model pumped storage (Note 7) or not? (0 = no, 1 = yes)
option_storage = int(parameters_general_values[np.where(parameters_general_list == 'option_storage', True, False)][0])

# [constant] density of water (kg/m^3) (introduced in eq. S3)
rho = parameters_general_values[np.where(parameters_general_list == 'rho', True, False)][0]

# [constant] gravitational acceleration (m/s^2) (introduced in eq. S8)
g = parameters_general_values[np.where(parameters_general_list == 'g', True, False)][0]

##### HYDROPOWER OPERATION PARAMETERS #####

# [set by user] threshold for determining whether HPP is "large" or "small" - if
# t_fill (eq. S1) is larger than threshold, classify as "large"
T_fill_thres = parameters_general_values[np.where(parameters_general_list == 'T_fill_thres', True, False)][0]

# [set by user] optional: requirement on Loss of Energy Expectation (criterion (ii) in Figure S1).
# As default, the HSW mix does not allow for any LOEE. However, this criterion could be relaxed.
# E.g. LOEE_allowed = 0.01 would mean that criterion (ii) is relaxed to 1% of yearly allowed LOEE instead of 0%.
LOEE_allowed = parameters_general_values[np.where(parameters_general_list == 'LOEE_allowed', True, False)][0]


# %% pre.3) Static parameters

# [set by user] name of hydropower plant
HPP_name = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name', True, False)][0].tolist()
HPP_name_data_inflow = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name_data_inflow', True, False)][0].tolist()
HPP_name_data_outflow_prescribed = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name_data_outflow_prescribed', True, False)][0].tolist()
HPP_name_data_CF_solar = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name_data_CF_solar', True, False)][0].tolist()
HPP_name_data_CF_wind = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name_data_CF_wind', True, False)][0].tolist()
HPP_name_data_evaporation = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name_data_evaporation', True, False)][0].tolist()
HPP_name_data_precipitation = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name_data_precipitation', True, False)][0].tolist()
HPP_name_data_load = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name_data_load', True, False)][0].tolist()
HPP_name_data_bathymetry = parameters_hydropower_values[np.where(parameters_hydropower_list == 'HPP_name_data_bathymetry', True, False)][0].tolist()

# [set by user] the parameter f_reg controls the fraction (<=1) of average inflow allocated to flexible use. Code B will enter a default if left empty by user
f_reg = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_reg', True, False)][0]

# [set by user] relative capacity of solar vs. wind to be installed (cf. explanation below eq. S13)
# e.g. c_solar_relative = 1 means only solar deployment, no wind deployment on the grid of each HPP in question.
c_solar_relative = parameters_hydropower_values[np.where(parameters_hydropower_list == 'c_solar_relative', True, False)][0]
c_wind_relative = 1 - c_solar_relative

# [set by user] maximum head (m)
h_max = parameters_hydropower_values[np.where(parameters_hydropower_list == 'h_max', True, False)][0]

# [set by user] maximum lake area (m^2)
A_max = parameters_hydropower_values[np.where(parameters_hydropower_list == 'A_max', True, False)][0]

# [set by user] maximum storage volume (m^3) and initial filling fraction
V_max = parameters_hydropower_values[np.where(parameters_hydropower_list == 'V_max', True, False)][0]
V_initial_frac = parameters_hydropower_values[np.where(parameters_hydropower_list == 'V_initial_frac', True, False)][0]

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
f_restart = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_restart', True, False)][0]

# [set by user] the parameter f_size controls allowed VRE overproduction and is the percentile value described in eq. S11
f_size = parameters_hydropower_values[np.where(parameters_hydropower_list == 'f_size', True, False)][0].astype(int)

# [set by user] number of turbines per hydropower plant (used for turbine use statistics in script C)
no_turbines = parameters_hydropower_values[np.where(parameters_hydropower_list == 'no_turbines', True, False)][0].astype(int)

# [set by user] percentile value used to calculate exceedance probability of delivered power
p_exceedance = parameters_hydropower_values[np.where(parameters_hydropower_list == 'p_exceedance', True, False)][0].astype(int)


# %% pre.4) Time series

# [preallocate]
L_norm = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))
evaporation_flux_hourly = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))
precipitation_flux_hourly = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))
Q_in_nat_hourly = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))
Q_out_stable_env_irr_hourly = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))
CF_solar_hourly = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))
CF_wind_hourly = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))

# [loop] over all HPP names to extract relevant time series from user-prepared Excel tables
for n in range(len(HPP_name)):
    
    # [set by user] Load curves (L_norm; see eq. S10)
    L_norm[:,:,n] = pd.read_excel (filename_load, sheet_name = HPP_name_data_load[n], header = None, usecols = range(column_start - 1, column_end), nrows = int(np.max(positions)))
    
    # [set by user] Precipitation and evaporation flux (kg/m^2/s)
    evaporation_flux_hourly[:,:,n] = pd.read_excel (filename_evaporation, sheet_name = HPP_name_data_evaporation[n], header = None, usecols = range(column_start - 1, column_end), nrows = int(np.max(positions)))
    precipitation_flux_hourly[:,:,n] = pd.read_excel (filename_precipitation, sheet_name = HPP_name_data_precipitation[n], header = None, usecols = range(column_start - 1, column_end), nrows = int(np.max(positions)))
    
    # [set by user] natural inflow and prescribed environmental/irrigation outflow at hourly timescale (m^3/s)
    Q_in_nat_hourly[:,:,n] = pd.read_excel (filename_inflow, sheet_name = HPP_name_data_inflow[n], header = None, usecols = range(column_start - 1, column_end), nrows = int(np.max(positions)))
    Q_out_stable_env_irr_hourly[:,:,n] = pd.read_excel (filename_outflow_prescribed, sheet_name = HPP_name_data_outflow_prescribed[n], header = None, usecols = range(column_start - 1, column_end), nrows = int(np.max(positions)))

    # [set by user] capacity factors weighted by location (eq. S12)
    CF_solar_hourly[:,:,n] = pd.read_excel (filename_CF_solar, sheet_name = HPP_name_data_CF_solar[n], header = None, usecols = range(column_start - 1, column_end), nrows = int(np.max(positions)))
    CF_wind_hourly[:,:,n] = pd.read_excel (filename_CF_wind, sheet_name = HPP_name_data_CF_wind[n], header = None, usecols = range(column_start - 1, column_end), nrows = int(np.max(positions)))
    


# %% pre.5) Bathymetry

# [preallocate]
temp_length_array = np.zeros(HPP_number)

# [loop] over all HPP bathymetries to find maximum length of necessary arrays
for n in range(len(HPP_name)):
    
    # [set by user] head-area-volume curves used during simulations
    temp = pd.read_excel (filename_bathymetry, sheet_name = HPP_name_data_bathymetry[n], header = None)
    temp_length_array[n] = len(temp)


# [preallocate]
calibrate_volume = np.full([int(np.max(temp_length_array)), HPP_number], np.nan)
calibrate_area = np.full([int(np.max(temp_length_array)), HPP_number], np.nan)
calibrate_head = np.full([int(np.max(temp_length_array)), HPP_number], np.nan)

# [loop] over all HPP names to extract relevant head-area-volume curves from user-prepared Excel files
for n in range(len(HPP_name)):
    
    # [set by user] head-area-volume curves used during simulations
    temp = pd.read_excel (filename_bathymetry, sheet_name = HPP_name_data_bathymetry[n], header = None)
    
    # [extract] volume (m^3)
    calibrate_volume[0:len(temp.iloc[:,0]),n] = temp.iloc[:,0]
    
    # [extract] area (m^2)
    calibrate_area[0:len(temp.iloc[:,1]),n] = temp.iloc[:,1]
    
    # [extract] head (m)
    calibrate_head[0:len(temp.iloc[:,2]),n] = temp.iloc[:,2]

