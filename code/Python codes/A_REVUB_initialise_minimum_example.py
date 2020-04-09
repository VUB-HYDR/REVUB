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
# All equation, section &c. numbers refer to that paper and its Supplementary Materials, unless otherwise mentioned.

import numpy as np
import pandas as pd
import numbers as nb

## pre.1) Time-related parameters

# [set by user] number of hydropower plants in this simulation
HPP_number = 2

# [set by user] The reference years used in the simulation
year_start = 1998
year_end = 2014
simulation_years = list(range(year_start, year_end + 1))

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

# [calculate] For each year in the simulation: determine if leap year or not;
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


## pre.2) Model parameters
        
##### GENERAL HYDROPOWER DATA #####
        
# [set by user] wish to model pumped storage (section S7) or not? (0 = no, 1 = yes)
option_storage = 1

# [constant] Density of water (kg/m^3) (introduced in eq. S3)
rho = 1000

# [constant] Gravitational acceleration (m/s^2) (introduced in eq. S8)
g = 9.81

##### HYDROPOWER OPERATION PARAMETERS #####

# [set by user] Turbine efficiency (introduced in eq. S8)
eta_turb = 0.8

# [set by user] Pumping efficiency
eta_pump = eta_turb

# [set by user] minimum required environmental outflow fraction (eq. S4, S5)
d_min = 0.4

# [set by user] alpha (eq. S6) for conventional HPP operation rule curve (eq. S4)
alpha = 2/3

# [set by user] gamma (eq. S4) for conventional HPP operation rule curve (eq. S4):
gamma_hydro = 10

# [set by user] f_opt (eq. S4, S5)
f_opt = 0.8

# [set by user] f_spill (eq. S7)
f_spill = 0.95

# [set by user] mu (eq. S7)
mu = 0.1

# [set by user] Thresholds f_stop and f_restart (see page 4) for stopping and restarting
# hydropower production to maintain minimum drawdown levels
f_stop = 0.10
f_restart = 0.20

# [set by user] Ramp rate restrictions (eq. S16, S37): fraction of full capacity per minute
dP_ramp_turb = 12.8/5/100
dP_ramp_pump = dP_ramp_turb

# [set by user] Array of C_{OR} values (eq. S14). The first value is the default. If the
# criterium on k_turb (eq. S28) is not met, the simulation is redone with the second value, &c.
C_OR_range_BAL = list(np.arange(1 - d_min, 0.05, -0.05))
C_OR_range_STOR = list(np.arange(1 - d_min, 0.05, -0.05))

# [set by user] Threshold for determining whether HPP is "large" or "small" - if
# t_fill (eq. S1) is larger than threshold, classify as "large"
T_fill_thres = 1.0

# [set by user] Optional: Requirement on Loss of Energy Expectation (criterion (ii) in Figure S1).
# As default, the HSW mix does not allow for any LOEE. However, this criterion could be relaxed.
# E.g. LOEE_allowed = 0.01 would mean that criterion (ii) is relaxed to 1% of yearly allowed LOEE instead of 0%.
LOEE_allowed = 0.00

# [set by user] The parameter f_size is the percentile value described in eq. S11
f_size = 90


## pre.3) Static parameters

# [set by user] name of hydropower plant
HPP_name = ["Bui", "Buyo"]

# [set by user] relative capacity of solar vs. wind to be installed (cf. explanation below eq. S13)
# e.g. c_solar_relative = 1 means only solar deployment, no wind deployment on the grid of each HPP in question.
c_solar_relative = np.array([0.573210768220617, 1])
c_wind_relative = 1 - c_solar_relative

# [set by user] maximum head (m)
h_max = np.array([80, 36.1])

# [set by user] maximum lake area (m^2)
A_max = np.array([4.4e8, 9e8])

# [set by user] maximum storage volume (m^3)
V_max = np.array([1.257e10, 8.3e9])

# [set by user] turbine capacity (MW)
P_r_turb = np.array([400, 165])

# [set by user] if using STOR scenario: lower reservoir capacity (MW)
V_lower_max = V_max/10**3

# [set by user] if using STOR scenario (only for Bui): pump capacity (MW)
P_r_pump = np.array([100, np.nan])

# [calculate] turbine and pump throughput (m^3/s, see explanation following eq. S8)
Q_max_turb = (P_r_turb) / (eta_turb*rho*g*h_max) * 10**6
Q_max_pump = (P_r_pump) / (eta_turb**(-1)*rho*g*h_max) * 10**6


## pre.4) Time series

# [preallocate]
L_norm = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))
evaporation_flux_hourly = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))
precipitation_flux_hourly = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))
Q_in_nat_hourly = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))
CF_solar_hourly = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))
CF_wind_hourly = np.zeros(shape = (int(np.max(positions)), len(simulation_years), HPP_number))

# [set by user] Load curves (L_norm; see eq. S10)
L_norm[:,:,0] = pd.read_excel (r'minimum_example_load.xlsx', sheet_name = 'GH', header = None)
L_norm[:,:,1] = pd.read_excel (r'minimum_example_load.xlsx', sheet_name = 'CIV', header = None)

# [set by user] Precipitation and evaporation flux (kg/m^2/s)
evaporation_flux_hourly[:,:,0] = pd.read_excel (r'minimum_example_evaporation.xlsx', sheet_name = 'Bui', header = None)
evaporation_flux_hourly[:,:,1] = pd.read_excel (r'minimum_example_evaporation.xlsx', sheet_name = 'Buyo', header = None)

precipitation_flux_hourly[:,:,0] = pd.read_excel (r'minimum_example_precipitation.xlsx', sheet_name = 'Bui', header = None)
precipitation_flux_hourly[:,:,1] = pd.read_excel (r'minimum_example_precipitation.xlsx', sheet_name = 'Buyo', header = None)

# [set by user] natural inflow at hourly timescale (m^3/s)
Q_in_nat_hourly[:,:,0] = pd.read_excel (r'minimum_example_inflow.xlsx', sheet_name = 'Bui', header = None)
Q_in_nat_hourly[:,:,1] = pd.read_excel (r'minimum_example_inflow.xlsx', sheet_name = 'Buyo', header = None)

# [set by user] capacity factors weighted by location (eq. S12)
CF_solar_hourly[:,:,0] = pd.read_excel (r'minimum_example_CF_solar.xlsx', sheet_name = 'GH', header = None)
CF_solar_hourly[:,:,1] = pd.read_excel (r'minimum_example_CF_solar.xlsx', sheet_name = 'CIV', header = None)

CF_wind_hourly[:,:,0] = pd.read_excel (r'minimum_example_CF_wind.xlsx', sheet_name = 'GH', header = None)
CF_wind_hourly[:,:,1] = pd.read_excel (r'minimum_example_CF_wind.xlsx', sheet_name = 'CIV', header = None)


## pre.5) Bathymetry

# [set by user] Calibration curves used during simulations
temp = pd.read_excel (r'minimum_example_bathymetry.xlsx', sheet_name = 'Bui', header = None)

# [preallocate]
calibrate_volume = np.full([len(temp.iloc[:,0]), HPP_number], np.nan)
calibrate_area = np.full([len(temp.iloc[:,0]), HPP_number], np.nan)
calibrate_head = np.full([len(temp.iloc[:,0]), HPP_number], np.nan)

# [extract] volume (m^3)
calibrate_volume[0:len(temp.iloc[:,0]),0] = temp.iloc[:,0]

# [extract] area (m^2)
calibrate_area[0:len(temp.iloc[:,1]),0] = temp.iloc[:,1]

# [extract] head (m)
calibrate_head[0:len(temp.iloc[:,2]),0] = temp.iloc[:,2]


# [set by user] Calibration curves used during simulations
temp = pd.read_excel (r'minimum_example_bathymetry.xlsx', sheet_name = 'Buyo', header = None)

# [extract] volume (m^3)
calibrate_volume[0:len(temp.iloc[:,0]),1] = temp.iloc[:,0]

# [extract] area (m^2)
calibrate_area[0:len(temp.iloc[:,1]),1] = temp.iloc[:,1]

# [extract] head (m)
calibrate_head[0:len(temp.iloc[:,2]),1] = temp.iloc[:,2]


del temp
