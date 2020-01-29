# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:12:57 2020

@author: ssterl
"""

# © 2019 CIREG project
# Author: Sebastian Sterl, Vrije Universiteit Brussel
# This code accompanies the paper "Smart renewable portfolios to displace fossil fuels and avoid hydropower overexploitation" by Sterl et al.
# All equation, section &c. numbers refer to that paper and its Supplementary Materials, unless otherwise mentioned.

import numpy as np
import pandas as pd
import numbers as nb

## REVUB.1) Set simulation accuracy

##### TECHNICAL SIMULATION PARAMETERS #####
# [set by user] This number defines the amount of discrete steps between 0 and max(E_hydro + E_solar + E_wind)
# reflecting the accuracy of determining the achieved ELCC
N_ELCC = 10**3

# [set by user] These values are used to get a good initial guess for the order of magnitude of the ELCC.
# This is done by multiplying them with yearly average E_{hydro}.
# A suitable range within which to identify the optimal solution (eq. S21) is thus obtained automatically
# for each HPP, regardless of differences in volume, head, rated power, &c.
# The value f_init_BAL_end may have to be increased in scenarios where the ELCC becomes extremely high,
# e.g. when extremely good balancing sources other than hydro are present.
# For the scenarios in (Sterl et al.), the below ranges work for all HPPs.
f_init_BAL_start = 0
f_init_BAL_step = 0.2
f_init_BAL_end = 3

# Idem for the optional STOR scenario
f_init_STOR_start = 0
f_init_STOR_step = 0.2
f_init_STOR_end = 3

# [set by user] Number of refinement loops for equilibrium search for min(Psi) (see eq. S21)
# Every +1 increases precision by one digit. Typically, 2 or 3 iterations suffice. 
# For (Sterl et al.), this parameter was set to 3.
N_refine_BAL = 2
N_refine_STOR = 2

# [set by user] When min(Psi) (eq. S21) is lower than this threshold, no further refinement loops
# are performed. This number can be increased to speed up the simulation.
psi_min_threshold = 0.00

# [set by user] Number of loops for iterative estimation of P_stable,BAL/STOR (see eq. S9 & explanation below eq. S19)
# Typically, 3-6 iterations suffice until convergence is achieved. For (Sterl et al.), this parameter was set to 6.
X_max_BAL = 3
X_max_STOR = 3


## REVUB.2) Preallocate variables for REVUB simulation


##### RESERVOIR INFLOW PARAMETERS #####

# [preallocate] Part of inflow that can be stored on annual timescales
Q_in_frac_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
Q_in_frac_store = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)

# [preallocate] Part of inflow that must be released to prevent overflowing (this term is by definition zero for large HPPs)
Q_in_RoR_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
Q_in_RoR_store = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)

# [preallocate] HPP category
HPP_category = ["" for x in range(HPP_number)]


##### HPP OPERATIONAL PARAMETERS #####


# [preallocate] Parameters tau_fill (eq. S1), phi (eq. S6), kappa (eq. S5) and f_reg (eq. S30) for each HPP
tau_fill = np.full([HPP_number], np.nan)
phi = np.full([HPP_number], np.nan)
kappa = np.full([HPP_number], np.nan)
f_reg = np.full([HPP_number], np.nan)

# [preallocate] Option for STOR scenario
STOR_break = np.zeros(shape = HPP_number)


##### RESERVOIR OUTFLOW PARAMETERS #####


# [preallocate] Various outflow data arrays (m^3/s) for CONV scenario (section S2, S3.1 and eq. S2)
Q_CONV_stable_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
Q_CONV_spill_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
Q_CONV_out_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)

# [preallocate] Various outflow data arrays (m^3/s) for BAL scenario (section S2, S3.2 and eq. S2)
Q_BAL_stable_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
Q_BAL_flexible_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
Q_BAL_spill_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
Q_BAL_out_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)

# [preallocate] Potential flexible outflow from eq. S17
Q_BAL_pot_turb_flexible = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)

# [preallocate] Various outflow data arrays (m^3/s) for optional STOR scenario (section S7)
Q_STOR_stable_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
Q_STOR_flexible_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
Q_STOR_pump_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
Q_STOR_out_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
Q_STOR_spill_hourly_upper = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
Q_STOR_spill_hourly_lower = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)

# [preallocate] Potential flexible outflow from eq. S17
Q_STOR_pot_turb_flexible = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)

# [preallocate] Potential pumped flow from eq. S38
Q_STOR_pot_pump_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)

# [preallocate] Outflow data (monthly mean) aggregated by month (m^3/s)
Q_CONV_out_monthly = np.zeros(shape = (months_yr, len(simulation_years), HPP_number))
Q_BAL_out_monthly = np.zeros(shape = (months_yr, len(simulation_years), HPP_number))
Q_STOR_out_monthly = np.zeros(shape = (months_yr, len(simulation_years), HPP_number))

# [preallocate] Outflow data (yearly mean) aggregated by year (m^3/s)
Q_BAL_out_yearly = np.zeros(shape = (len(simulation_years), HPP_number))
Q_STOR_out_yearly = np.zeros(shape = (len(simulation_years), HPP_number))

# [preallocate] Data arrays (monthly mean) aggregated by month, accounting for the simulated actual use
# of inflow for reservoir filling or direct discharge, depending on water levels (see text below eq. S33) (m^3/s)
Q_used_as_nonRoR_monthly = np.zeros(shape = (months_yr, len(simulation_years), HPP_number))
Q_used_as_RoR_monthly = np.zeros(shape = (months_yr, len(simulation_years), HPP_number))


##### RESERVOIR VOLUME #####


# [preallocate] Reservoir volume data (in m^3)
V_CONV_hourly = np.full([int(np.max(positions)) + 1, len(simulation_years), HPP_number], np.nan)
V_BAL_hourly = np.full([int(np.max(positions)) + 1, len(simulation_years), HPP_number], np.nan)
V_STOR_hourly_upper = np.full([int(np.max(positions)) + 1, len(simulation_years), HPP_number], np.nan)
V_STOR_hourly_lower = np.full([int(np.max(positions)) + 1, len(simulation_years), HPP_number], np.nan)

# [preallocate] The same data as 1D-array (full time series from start to end, not ordered by year)
V_CONV_series_hourly = np.full([int(sum(hrs_byyear)), HPP_number], np.nan)
V_BAL_series_hourly = np.full([int(sum(hrs_byyear)), HPP_number], np.nan)
V_STOR_series_hourly_upper = np.full([int(sum(hrs_byyear)), HPP_number], np.nan)
V_STOR_series_hourly_lower = np.full([int(sum(hrs_byyear)), HPP_number], np.nan)


##### RESERVOIR LAKE AREA #####


# [preallocate] Lake surface area (in m^2)
A_CONV_hourly = np.full([int(np.max(positions)) + 1, len(simulation_years), HPP_number], np.nan)
A_BAL_hourly = np.full([int(np.max(positions)) + 1, len(simulation_years), HPP_number], np.nan)
A_STOR_hourly_upper = np.full([int(np.max(positions)) + 1, len(simulation_years), HPP_number], np.nan)

# [preallocate] The same data as 1D-array (full time series from start to end, not ordered by year)
A_CONV_series_hourly = np.full([int(sum(hrs_byyear)), HPP_number], np.nan)
A_BAL_series_hourly = np.full([int(sum(hrs_byyear)), HPP_number], np.nan)
A_STOR_series_hourly_upper = np.full([int(sum(hrs_byyear)), HPP_number], np.nan)


##### RESERVOIR WATER LEVEL / HYDRAULIC HEAD #####


# [preallocate] hydraulic head from water level to turbine (in m)
h_CONV_hourly = np.full([int(np.max(positions)) + 1, len(simulation_years), HPP_number], np.nan)
h_BAL_hourly = np.full([int(np.max(positions)) + 1, len(simulation_years), HPP_number], np.nan)
h_STOR_hourly = np.full([int(np.max(positions)) + 1, len(simulation_years), HPP_number], np.nan)

# [preallocate] The same data as 1D-array (full time series from start to end, not ordered by year)
h_CONV_series_hourly = np.full([int(sum(hrs_byyear)), HPP_number], np.nan)
h_BAL_series_hourly = np.full([int(sum(hrs_byyear)), HPP_number], np.nan)
h_STOR_series_hourly = np.full([int(sum(hrs_byyear)), HPP_number], np.nan)


##### SOLAR AND WIND CAPACITY #####


# [preallocate] Yearly power generation by each MW of solar+wind capacity with solar-wind ratio
# given by c_solar: c_wind (needed to calculate multipliers, below)
E_SW_per_MW_BAL_yearly = np.zeros(shape = (len(simulation_years), HPP_number))
E_SW_per_MW_STOR_yearly = np.zeros(shape = (len(simulation_years), HPP_number))

# [preallocate] Capacity multiplier (MW; the product c_solar_relative*c_multiplier [see later]
# equals the factor c_solar in eq. S9; idem for wind)
c_multiplier_BAL = np.zeros(shape = (len(simulation_years), HPP_number))
c_multiplier_STOR = np.zeros(shape = (len(simulation_years), HPP_number))


##### POWER GENERATION PARAMETERS: HYDRO #####


# [preallocate] Hydropower generation in CONV scenario (in MW or MWh/h)
P_CONV_hydro_stable_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
P_CONV_hydro_RoR_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)

# [preallocate] Hydropower generation in BAL scenario (in MW or MWh/h)
P_BAL_hydro_stable_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
P_BAL_hydro_flexible_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
P_BAL_hydro_RoR_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)

# [preallocate] Hydropower generation/storage in optional STOR scenario (in MW or MWh/h)
P_STOR_hydro_stable_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
P_STOR_pump_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
P_STOR_hydro_flexible_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan);

# [preallocate] Maximum possible power output after accounting for ramp rate restrictions
# (in MW or MWh/h, see eq. S16, S37)
P_BAL_ramp_restr_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
P_STOR_ramp_restr_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
P_STOR_ramp_restr_pump_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)

# [preallocate] Turbine utilization rate (fraction; see eq. S28)
k_turb_hourly_BAL = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
k_turb_hourly_STOR = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)


##### POWER GENERATION PARAMETERS: SOLAR & WIND #####

# [preallocate] Power generation from solar and wind power (MW or MWh/h)
P_BAL_solar_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
P_BAL_wind_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
P_STOR_solar_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
P_STOR_wind_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)


##### POWER GENERATION PARAMETERS: HYDRO-SOLAR-WIND #####


# [preallocate] Load difference (eq. S9; in MW or MWh/h)
P_BAL_difference_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
P_STOR_difference_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)

# [preallocate] ELCC (Effective Load Carrying Capability = optimal series L(t) in eq. S10; in MW or MWh/h)
L_followed_BAL_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
L_followed_STOR_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)

# [preallocate] P_inflexible (stable hydro + solar + wind in eq. S9; in MW or MWh/h)
P_BAL_inflexible_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
P_STOR_inflexible_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)

# [preallocate] Binary variable [0 or 1] determining whether hydropower plant is operating (1)
# or shut off in case of extreme drought (0) (see section S3.1 and S8)
hydro_CONV_curtailment_factor_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
hydro_BAL_curtailment_factor_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
hydro_STOR_curtailment_factor_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)


##### LOAD PROFILE DATA #####


# [preallocate] Load curve L(t) from eq. S9 (MW or MWh/h)
L_BAL_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
L_STOR_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)

# [preallocate] Optimal E_solar + E_wind identified when looping over a range of possible ELCC
# values to identify min(Psi) (eq. S21; in MWh/year)
E_SW_loop_BAL_opt = np.zeros(shape = HPP_number)
E_SW_loop_STOR_opt = np.zeros(shape = HPP_number)


##### YEARLY ELECTRICITY GENERATION PARAMETERS (SIMULATION OUTCOMES) #####


# [preallocate] Hydropower generation in CONV (MWh/year)
E_hydro_CONV_stable_yearly = np.zeros(shape = (len(simulation_years), HPP_number))
E_hydro_CONV_RoR_yearly = np.zeros(shape = (len(simulation_years), HPP_number))

# [preallocate] Hydropower generation in BAL (MWh/year) (eq. S24, S33)
E_hydro_BAL_stable_yearly = np.zeros(shape = (len(simulation_years), HPP_number))
E_hydro_BAL_flexible_yearly = np.zeros(shape = (len(simulation_years), HPP_number))
E_hydro_BAL_nonRoR_yearly = np.zeros(shape = (len(simulation_years), HPP_number))
E_hydro_BAL_RoR_yearly = np.zeros(shape = (len(simulation_years), HPP_number))

# [preallocate] Hydropower generation in STOR (MWh/year) (eq. S24, S33)
E_hydro_STOR_stable_yearly = np.zeros(shape = (len(simulation_years), HPP_number))
E_hydro_STOR_flexible_yearly = np.zeros(shape = (len(simulation_years), HPP_number))
E_hydro_STOR_yearly = np.zeros(shape = (len(simulation_years), HPP_number))
E_hydro_STOR_pump_yearly = np.zeros(shape = (len(simulation_years), HPP_number))

# [preallocate] Solar and wind power generation in BAL (MWh/year) (eq. S25)
E_solar_BAL_yearly = np.zeros(shape = (len(simulation_years), HPP_number))
E_wind_BAL_yearly = np.zeros(shape = (len(simulation_years), HPP_number))

# [preallocate] Hydro + solar + wind generation in BAL (MWh/year)
E_SW_BAL_yearly = np.zeros(shape = (len(simulation_years), HPP_number))                     # total yearly SW generation
E_HSW_BAL_yearly = np.full([len(simulation_years), HPP_number], np.nan)                     # total yearly HSW generation
E_overproduced_BAL_yearly = np.zeros(shape = (len(simulation_years), HPP_number))           # yearly overproduction compared to ELCC in BAL (MWh/year)
E_SW_BAL_without_overproduction = np.zeros(shape = (len(simulation_years), HPP_number))     # amount of SW generation minus overproduction in BAL (MWh/year)
share_SW_BAL = np.zeros(shape = (len(simulation_years), HPP_number))                        # share of solar-to-wind in power generation in BAL (% solar)

# [preallocate] Solar and wind power generation in STOR (MWh/year) (eq. S25)
E_solar_STOR_yearly = np.zeros(shape = (len(simulation_years), HPP_number))
E_wind_STOR_yearly = np.zeros(shape = (len(simulation_years), HPP_number))

# [preallocate] Hydro + solar + wind generation in STOR (MWh/year)
E_SW_STOR_yearly = np.zeros(shape = (len(simulation_years), HPP_number))                    # total yearly SW generation
E_HSW_STOR_yearly = np.full([len(simulation_years), HPP_number], np.nan)                    # total yearly HSW generation
E_overproduced_STOR_yearly = np.zeros(shape = (len(simulation_years), HPP_number))          # yearly overproduction compared to ELCC in STOR (MWh/year)
E_SW_STOR_without_overproduction = np.zeros(shape = (len(simulation_years), HPP_number))    # amount of SW generation minus overproduction in STOR (MWh/year)
share_SW_STOR = np.zeros(shape = (len(simulation_years), HPP_number))                       # share of solar-to-wind in power generation in STOR (% solar)


##### IDENTIFYING THE ACHIEVED ELCC UNDER OPTIMAL HSW COMBINATION #####


# [preallocate] Range of ELCCs from which to identify the actual one post-simulation
# (MWh/year). Accuracy is given by the parameter N_ELCC (amount of discrete values).
P_followed_BAL_range = np.zeros(shape = (len(simulation_years), N_ELCC, HPP_number))
P_followed_STOR_range = np.zeros(shape = (len(simulation_years), N_ELCC, HPP_number))

# [preallocate] Index of achieved ELCC in the above range
P_followed_BAL_index = np.zeros(shape = (len(simulation_years), HPP_number))
P_followed_STOR_index = np.zeros(shape = (len(simulation_years), HPP_number))

# [preallocate] Achieved ELCC (MWh/year)
ELCC_CONV_yearly = np.zeros(shape = (len(simulation_years), HPP_number))
ELCC_BAL_yearly = np.zeros(shape = (len(simulation_years), HPP_number))
ELCC_STOR_yearly = np.zeros(shape = (len(simulation_years), HPP_number))


##### STATISTICAL OUTCOMES OF POWER GENERATION AND ELCC #####

# [preallocate] Hydropower generation under CONV - statistics (GWh/year)
E_hydro_CONV_stable_statistics_median = np.full(HPP_number, np.nan)
E_hydro_CONV_RoR_statistics_median = np.full(HPP_number, np.nan)

# [preallocate] Hydropower generation under BAL - statistics(GWh/year)
E_hydro_BAL_nonRoR_statistics_median = np.full(HPP_number, np.nan)
E_hydro_BAL_RoR_statistics_median = np.full(HPP_number, np.nan)

# [preallocate] Hydropower generation under STOR - statistics (GWh/year)
# [note: HPPs with RoR outflow components are not considered for STOR]
E_hydro_STOR_statistics_median = np.full(HPP_number, np.nan)

# [preallocate] HSW generation under BAL - statistics (GWh/year)
E_HSW_BAL_statistics_median = np.full(HPP_number, np.nan)
E_HSW_BAL_statistics_pct25 = np.full(HPP_number, np.nan)
E_HSW_BAL_statistics_pct75 = np.full(HPP_number, np.nan)
E_solar_BAL_statistics_median = np.full(HPP_number, np.nan)
E_solar_BAL_statistics_pct25 = np.full(HPP_number, np.nan)
E_solar_BAL_statistics_pct75 = np.full(HPP_number, np.nan)
E_wind_BAL_statistics_median = np.full(HPP_number, np.nan)
E_wind_BAL_statistics_pct25 = np.full(HPP_number, np.nan)
E_wind_BAL_statistics_pct75 = np.full(HPP_number, np.nan)

# [preallocate] HSW generation under STOR - statistics (GWh/year)
E_HSW_STOR_statistics_median = np.full(HPP_number, np.nan)
E_HSW_STOR_statistics_pct25 = np.full(HPP_number, np.nan)
E_HSW_STOR_statistics_pct75 = np.full(HPP_number, np.nan)
E_solar_STOR_statistics_median = np.full(HPP_number, np.nan)
E_solar_STOR_statistics_pct25 = np.full(HPP_number, np.nan)
E_solar_STOR_statistics_pct75 = np.full(HPP_number, np.nan)
E_wind_STOR_statistics_median = np.full(HPP_number, np.nan)
E_wind_STOR_statistics_pct25 = np.full(HPP_number, np.nan)
E_wind_STOR_statistics_pct75 = np.full(HPP_number, np.nan)

# [preallocate] Achieved ELCC - statistics (GWh/year)
ELCC_CONV_statistics_median = np.full(HPP_number, np.nan)
ELCC_CONV_statistics_pct25 = np.full(HPP_number, np.nan)
ELCC_CONV_statistics_pct75 = np.full(HPP_number, np.nan)
ELCC_BAL_statistics_median = np.full(HPP_number, np.nan)
ELCC_BAL_statistics_pct25 = np.full(HPP_number, np.nan)
ELCC_BAL_statistics_pct75 = np.full(HPP_number, np.nan)
ELCC_STOR_statistics_median = np.full(HPP_number, np.nan)
ELCC_STOR_statistics_pct25 = np.full(HPP_number, np.nan)
ELCC_STOR_statistics_pct75 = np.full(HPP_number, np.nan)

# [preallocate] Ratio of ELCC to total hydropower generation (see Table B1).
# This is a measure of how "good" the HPP is at supporting SW in the given SW mix
ratio_ELCC_E_hydro_BAL_yearly = np.full([len(simulation_years), HPP_number], np.nan)
ratio_ELCC_E_hydro_BAL_median = np.full(HPP_number, np.nan)
ratio_ELCC_E_hydro_STOR_yearly = np.full([len(simulation_years), HPP_number], np.nan)
ratio_ELCC_E_hydro_STOR_median = np.full(HPP_number, np.nan)

# [preallocate] Yearly average capacity factor of HPP turbines (%)
CF_hydro_CONV_yearly = np.full([len(simulation_years), HPP_number], np.nan)

# [preallocate] Hourly capacity factor for BAL and STOR scenario (%)
CF_hydro_BAL_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
CF_hydro_STOR_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)

# [preallocate] RLDC = Residual Load Duration Curve; sorted array of P_stable +
# P_flexible + P_solar + P_wind (- P_pump) (in MW or MWh/h)
L_res_BAL_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)
L_res_STOR_hourly = np.full([int(np.max(positions)), len(simulation_years), HPP_number], np.nan)

# [preallocate] Fraction of ELCC unmet by HSW operation. note: as long as the parameter
# "LOEE_allowed" is set to zero, these arrays will be zero.
# If LOEE_allowed > 0, these arrays will indicate how the allowed unmet fraction of the ELCC
# is distributed over different months.
L_unmet_BAL_frac_bymonth = np.zeros(shape = (months_yr, len(simulation_years), HPP_number))
L_unmet_STOR_frac_bymonth = np.zeros(shape = (months_yr, len(simulation_years), HPP_number))


##### DATA OUTPUT FILE #####


# [preallocate] Table with main output parameters for post-processing (column 7-12 in Table S3-7)
data_SI_B_BAL = np.full([HPP_number, 7], np.nan)
data_SI_B_STOR = np.full([HPP_number, 7], np.nan)



## REVUB.3) Classify HPPs


# [loop] to classify all HPPs
for HPP in range(HPP_number):
    
    # [calculate] f_reg (eq. S29, S30 - solution for f_reg of t_fill,frac = 1 in eq. S29)
    f_reg[HPP] = (V_max[HPP]/(min(np.sum(days_year,0))*hrs_day*secs_hr*T_fill_thres))/np.nanmean(Q_in_nat_hourly[:,:,HPP])
    
    # [calculate] Determine dam category based on f_reg (section A5)
    # Here "large" HPPs are designated by "A", "small" HPPs by "B".
    
    if f_reg[HPP] < 1:
        # [define] as small HPP
        HPP_category[HPP] = "B"
        # [calculate] flexibly usable inflow for small HPPs (eq. S30)
        Q_in_frac_hourly[:,:,HPP] = f_reg[HPP]*Q_in_nat_hourly[:,:,HPP]
    else:
        # [define] as large HPP
        HPP_category[HPP] = "A"
        # [calculate] all flow can be used flexibly for large HPPs
        Q_in_frac_hourly[:,:,HPP] = Q_in_nat_hourly[:,:,HPP]
        
    # [calculate] the component Q_RoR for small HPPs (section A5)
    Q_in_RoR_hourly[:,:,HPP] = Q_in_nat_hourly[:,:,HPP] - Q_in_frac_hourly[:,:,HPP]
    
    ##### SPECIFY OUTFLOW CURVE (CONV) #####
    # [calculate] tau_fill (eq. S1) for each HPP
    tau_fill[HPP] = (np.nanmean(Q_in_frac_hourly[:,:,HPP] * (min(np.sum(days_year,0))*hrs_day*secs_hr)/V_max[HPP]))**(-1)
    
    # [calculate] phi (eq. S6) for each HPP
    phi[HPP] = alpha*np.sqrt(tau_fill[HPP])
    
    # [calculate] kappa (eq. S5) for each HPP
    kappa[HPP] = 1/(f_opt**phi[HPP])*((np.exp(1))**(1 - d_min) - 1)
    
    # [initialize] store Q_in_frac_hourly and Q_in_RoR_hourly; these may change during the simulations
    # but need to be reinitialized for every iteration step (e.g. every new c_solar, c_wind in eq. S9)
    Q_in_frac_store[:,:,HPP] = Q_in_frac_hourly[:,:,HPP]
    Q_in_RoR_store[:,:,HPP] = Q_in_RoR_hourly[:,:,HPP]
    


## REVUB.4) Core REVUB simulation

# This section carries out the actual REVUB optimization.

# [loop] carry out CONV, BAL and (optionally) STOR simulation for every HPP
for HPP in range(HPP_number):
    
    # [display] HPP for which simulation is being performed
    print("HPP", HPP + 1, "/", HPP_number, ":", HPP_name[HPP])
    
    
    ###############################################################
    ############----------- CONV simulation -----------############
    ###############################################################
    
    # [initialize] ensure Q_in_nat_flex and Q_in_nat_out are written correctly at the beginning of each simulation
    Q_in_frac_hourly[:,:,HPP] = Q_in_frac_store[:,:,HPP]
    Q_in_RoR_hourly[:,:,HPP] = Q_in_RoR_store[:,:,HPP]
    
    # [initialize] Calculate multiannual average flow for conventional operating rules (eq. S4)
    Q_in_nat_av = np.nanmean(Q_in_frac_hourly[:,:,HPP])
    
    # [initialize] This variable is equal to unity by default, but set to zero in case of extreme droughts forcing a
    # temporary curtailment on hydropower generation (section S3.1)
    hydro_CONV_curtailment_factor_hourly[:,:,HPP] = 1
    
    # [display] CONV simulation underway
    print("(i) simulating CONV")
    
    # [loop] across all simulation years
    for y in range(len(simulation_years)):
    
        # [read] vector with hours in each year
        hrs_year = range(int(hrs_byyear[y]))
                
        # [initialize] initial values of volume (m^3), area (m^2) and hydraulic head (m) for each simulation year
        if y == 0:
            
            V_CONV_hourly[0,y,HPP] = V_max[HPP]*f_opt
            h_temp = np.where(abs(calibrate_volume[:,HPP] - V_CONV_hourly[0,y,HPP]) == min(abs(calibrate_volume[:,HPP] - V_CONV_hourly[0,y,HPP])))[0][0]
            A_CONV_hourly[0,y,HPP] = calibrate_area[h_temp,HPP]
            h_CONV_hourly[0,y,HPP] = calibrate_head[h_temp,HPP]
            
        else:
            
            temp = V_CONV_hourly[:,y-1,HPP]
            temp = temp[np.isfinite(temp)]
            V_CONV_hourly[0,y,HPP] = temp[-1]
            
            temp = A_CONV_hourly[:,y-1,HPP]
            temp = temp[np.isfinite(temp)]
            A_CONV_hourly[0,y,HPP] = temp[-1]
            
            temp = h_CONV_hourly[:,y-1,HPP]
            temp = temp[np.isfinite(temp)]
            h_CONV_hourly[0,y,HPP] = temp[-1]
            
            
        # [loop] over all time steps in each simulation year to calculate reservoir dynamics and hydropower generation
        for n in hrs_year:
            
            # [calculate] stable outflow Q_stable in m^3/s according to conventional management (eq. S4)
            if V_CONV_hourly[n,y,HPP]/V_max[HPP] < f_opt:
                
                Q_CONV_stable_hourly[n,y,HPP] = (d_min + np.log(kappa[HPP]*(V_CONV_hourly[n,y,HPP]/V_max[HPP])**phi[HPP] + 1))*Q_in_nat_av
                Q_CONV_spill_hourly[n,y,HPP] = 0
                
            elif V_CONV_hourly[n,y,HPP]/V_max[HPP] < f_spill:
                
                Q_CONV_stable_hourly[n,y,HPP] = (np.exp(gamma_hydro*(V_CONV_hourly[n,y,HPP]/V_max[HPP] - f_opt)**2))*Q_in_nat_av
                Q_CONV_spill_hourly[n,y,HPP] = 0
                
            else:
                
                # [calculate] spilling component (eq. S7)
                Q_CONV_stable_hourly[n,y,HPP] = (np.exp(gamma_hydro*(V_CONV_hourly[n,y,HPP]/V_max[HPP] - f_opt)**2))*Q_in_nat_av
                Q_CONV_spill_hourly[n,y,HPP] = (Q_in_frac_hourly[n,y,HPP] + (precipitation_flux_hourly[n,y,HPP] - evaporation_flux_hourly[n,y,HPP])*A_CONV_hourly[n,y,HPP]/rho)*(1 + mu) - Q_CONV_stable_hourly[n,y,HPP]

                # [check] spilling component cannot be negative (eq. S7)
                if Q_CONV_spill_hourly[n,y,HPP] < 0:
                    Q_CONV_spill_hourly[n,y,HPP] = 0
            
            
            # [check] stable outflow is reduced to zero in case of droughts
            Q_CONV_stable_hourly[n,y,HPP] = Q_CONV_stable_hourly[n,y,HPP] * hydro_CONV_curtailment_factor_hourly[n,y,HPP]
            
            # [calculate] total net outflow in m^3/s (eq. S2)
            Q_CONV_out_hourly[n,y,HPP] = Q_CONV_stable_hourly[n,y,HPP] + Q_CONV_spill_hourly[n,y,HPP] + Q_in_RoR_hourly[n,y,HPP]
            
            # [calculate] hydropower generation in MW (eq. S8)
            Q_pot_turb_CONV = np.min([Q_CONV_stable_hourly[n,y,HPP], Q_max_turb[HPP]])
            P_CONV_hydro_stable_hourly[n,y,HPP] = Q_pot_turb_CONV*eta_turb*rho*g*h_CONV_hourly[n,y,HPP]/10**6
            
            # [calculate] hydropower generation from RoR flow component in MW (eq. S32)
            P_CONV_hydro_RoR_hourly[n,y,HPP] = np.min([Q_in_RoR_hourly[n,y,HPP], np.max([0, Q_max_turb[HPP] - Q_CONV_stable_hourly[n,y,HPP]]) ])*eta_turb*rho*g*h_CONV_hourly[n,y,HPP]/10**6
            
            # [calculate] reservoir volume in m^3 at next time step (eq. S3, S31)
            V_CONV_hourly[n+1,y,HPP] = V_CONV_hourly[n,y,HPP] + (Q_in_frac_hourly[n,y,HPP] - Q_CONV_stable_hourly[n,y,HPP] - Q_CONV_spill_hourly[n,y,HPP] + (precipitation_flux_hourly[n,y,HPP] - evaporation_flux_hourly[n,y,HPP])*A_CONV_hourly[n,y,HPP]/rho)*secs_hr
            
            # [calculate] reservoir lake area in m^2 and hydraulic head in m from bathymetric relationship
            h_temp = np.where(abs(calibrate_volume[:,HPP] - V_CONV_hourly[n+1,y,HPP]) == min(abs(calibrate_volume[:,HPP] - V_CONV_hourly[n+1,y,HPP])))[0][0]
            A_CONV_hourly[n+1,y,HPP] = calibrate_area[h_temp,HPP]
            h_CONV_hourly[n+1,y,HPP] = calibrate_head[h_temp,HPP]
            
            # [calculate] whether lake levels have dropped so low as to require hydropower curtailment
            # [calculate] for small HPPs: use "RoR" flow component to fill up reservoir in case water levels have dropped below f_release*V_max
            # (see explanation below eq. S33)
            if HPP_category[HPP] == "B":
                if V_CONV_hourly[n+1,y,HPP] < f_restart*V_max[HPP]:
                    if n < len(hrs_year) - 1:
                        Q_in_frac_hourly[n+1,y,HPP] = Q_in_frac_hourly[n+1,y,HPP] + Q_in_RoR_hourly[n+1,y,HPP]
                        Q_in_RoR_hourly[n+1,y,HPP] = 0
                    elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                        Q_in_frac_hourly[0,y+1,HPP] = Q_in_frac_hourly[0,y+1,HPP] + Q_in_RoR_hourly[0,y+1,HPP]
                        Q_in_RoR_hourly[0,y+1,HPP] = 0
                        
            # [calculate] for large and small HPPs: curtail hydropower generation in case water levels have dropped below f_stop*V_max
            # (see section S3.1)
            if V_CONV_hourly[n+1,y,HPP] < f_stop*V_max[HPP]:
                if n < len(hrs_year) - 1:
                    hydro_CONV_curtailment_factor_hourly[n+1,y,HPP] = 0
                elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                    hydro_CONV_curtailment_factor_hourly[0,y+1,HPP] = 0
            
            
            # [calculate] restart hydropower generation if reservoir levels have recovered
            # (see section S3.1)
            if hydro_CONV_curtailment_factor_hourly[n,y,HPP] == 0 and V_CONV_hourly[n+1,y,HPP] > f_restart*V_max[HPP]:
                if n < len(hrs_year) - 1:
                    hydro_CONV_curtailment_factor_hourly[n+1,y,HPP] = 1
                elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                    hydro_CONV_curtailment_factor_hourly[0,y+1,HPP] = 1
            
            elif hydro_CONV_curtailment_factor_hourly[n,y,HPP] == 0 and V_CONV_hourly[n+1,y,HPP] <= f_restart*V_max[HPP]:
                if n < len(hrs_year) - 1:
                    hydro_CONV_curtailment_factor_hourly[n+1,y,HPP] = 0
                elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                    hydro_CONV_curtailment_factor_hourly[0,y+1,HPP] = 0
            
            # [calculate] total hydropower generation in MWh/year (eq. S24)
            E_hydro_CONV_stable_yearly[y,HPP] = np.sum(P_CONV_hydro_stable_hourly[hrs_year,y,HPP])
            E_hydro_CONV_RoR_yearly[y,HPP] = np.sum(P_CONV_hydro_RoR_hourly[hrs_year,y,HPP])
            
            # [arrange] aggregate hourly outflow by month
            for m in range(months_yr):
                Q_CONV_out_monthly[m,y,HPP] = np.mean(Q_CONV_out_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP])
                
    
    # [arrange] complete time series of water volume, area and levels
    for y in range(len(simulation_years)): 
        V_CONV_hourly[int(hrs_byyear[y]),y,HPP] = np.nan
        A_CONV_hourly[int(hrs_byyear[y]),y,HPP] = np.nan
        h_CONV_hourly[int(hrs_byyear[y]),y,HPP] = np.nan
        
    temp_volume_upper_CONV_series = V_CONV_hourly[:,:,HPP]
    temp_volume_upper_CONV_series = (np.transpose(temp_volume_upper_CONV_series)).ravel()
    temp_volume_upper_CONV_series = temp_volume_upper_CONV_series[np.isfinite(temp_volume_upper_CONV_series)]
    V_CONV_series_hourly[:,HPP] = temp_volume_upper_CONV_series
    
    temp_area_CONV_series = A_CONV_hourly[:,:,HPP]
    temp_area_CONV_series = (np.transpose(temp_area_CONV_series)).ravel()
    temp_area_CONV_series = temp_area_CONV_series[np.isfinite(temp_area_CONV_series)]
    A_CONV_series_hourly[:,HPP] = temp_area_CONV_series
    
    temp_head_CONV_series = h_CONV_hourly[:,:,HPP]
    temp_head_CONV_series = (np.transpose(temp_head_CONV_series)).ravel()
    temp_head_CONV_series = temp_head_CONV_series[np.isfinite(temp_head_CONV_series)]
    h_CONV_series_hourly[:,HPP] = temp_head_CONV_series

    
    # [display] once CONV simulation is complete
    print("done")
    
    
    ###############################################################
    ############------------ BAL iterations -----------############
    ###############################################################
    
    # [display] start of iterations to find optimal solution for BAL operation
    print("(ii) finding optimal BAL solution")
    
    # [loop] with incrementally increased C_OR values, starting at C_OR = 1 - d_min (section S4)
    for q in range(len(C_OR_range_BAL)):
        
        # [calculate] ratio of stable (environmental) to average total outflow (see eq. S14)
        Q_stable_ratio = 1 - C_OR_range_BAL[q]

        # [display] refinement step in BAL simulation
        print("C_OR = ", np.round(100*C_OR_range_BAL[q], decimals = 1), "%")
        
        # [loop] across refinement steps to increase accuracy
        for n_refine_BAL in range(N_refine_BAL):
            
            # [initialize] range for current refinement step; each step increases accuracy by one digit
            if n_refine_BAL == 0:
                f_demand_BAL_start = f_init_BAL_start
                f_demand_BAL_step = f_init_BAL_step
                f_demand_BAL_end = f_init_BAL_end
            elif n_refine_BAL > 0:
                f_demand_BAL_start = f_demand_opt_BAL - f_demand_BAL_step
                f_demand_BAL_end = f_demand_opt_BAL + f_demand_BAL_step
                f_demand_BAL_step = f_demand_BAL_step/10
            
            f_demand_BAL = np.arange(f_demand_BAL_start, f_demand_BAL_end + f_demand_BAL_step, f_demand_BAL_step)

            # [preallocate] psi (eq. S21)
            psi_BAL = np.full([len(f_demand_BAL)], np.nan)
            
            # [loop] to find optimal values of E_solar and E_wind (eq. S25) by locating minimum in psi (eq. S22)
            for f in range(len(f_demand_BAL)):
                
                # [display] progress within each refinement step in BAL simulation
                print("refinement step", n_refine_BAL + 1, "/", N_refine_BAL, "> scanning:", np.floor(100*(f + 1)/len(f_demand_BAL)), "%")

                # [initialize] realistic value of total SW power (MWh/year) so that we can loop over realistic values of c_solar and c_wind (eq. S25)
                E_SW_loop_BAL = np.mean(E_hydro_CONV_stable_yearly[:,HPP])*f_demand_BAL[f]*np.ones(shape = (len(E_hydro_CONV_stable_yearly[:,HPP])))
                
                # [preallocate] stable hydropower generation P_stable in MW (see explanation below eq. S19)
                P_BAL_hydro_stable_hourly[:,:,HPP] = np.nan
                
                # [loop] across all simulation years to identify realistic c_solar and c_wind values
                for y in range(len(simulation_years)):

                    # [read] vector with hours in each year
                    hrs_year = range(int(hrs_byyear[y]))
                    
                    # [calculate] determine realistic amount of SW capacity in MW (c_solar, c_wind) corresponding to generation equal to E_SW_loop_BAL
                    E_SW_per_MW_BAL_yearly[y,HPP] = np.sum(c_solar_relative[HPP]*CF_solar_hourly[hrs_year,y,HPP] + c_wind_relative[HPP]*CF_wind_hourly[hrs_year,y,HPP])
                    c_multiplier_BAL[y,HPP] = E_SW_loop_BAL[y]/E_SW_per_MW_BAL_yearly[y,HPP]
                    
                # [loop] perform iterations to get converged estimate of P_stable (see explanation below eq. S19)
                for x in range(X_max_BAL):
                                    
                    # [calculate] environmentally required outflow (eq. S14)
                    temp_Q_out_BAL = Q_in_nat_av*np.ones(shape = (len(Q_CONV_stable_hourly),len(Q_CONV_stable_hourly[0])))
                    temp_Q_out_BAL[np.isnan(Q_CONV_stable_hourly[:,:,HPP])] = np.nan
                    Q_BAL_stable_hourly[:,:,HPP] = Q_stable_ratio*temp_Q_out_BAL
                    
                    # [initialize] ensure Q_in_frac_hourly and Q_in_RoR_hourly are written correctly at the beginning of each step in the loop
                    Q_in_frac_hourly[:,:,HPP] = Q_in_frac_store[:,:,HPP]
                    Q_in_RoR_hourly[:,:,HPP] = Q_in_RoR_store[:,:,HPP]
                    
                    # [initialize] This variable is equal to unity by default, but set to zero in case of extreme droughts forcing a
                    # temporary curtailment on hydropower generation (section S3.1)
                    hydro_BAL_curtailment_factor_hourly[:,:,HPP] = 1

                    # [loop] across all simulation years to initialize P_stable (see explanation below eq. S19)
                    for y in range(len(simulation_years)):
                    
                        # [read] vector with hours in each year
                        hrs_year = range(int(hrs_byyear[y]))

                        # [initialize] stable hydropower generation P_stable in MW (see explanation below eq. S19)
                        # use estimate P_stable,BAL = (1 - C_OR)*P_stable,CONV as initial guess
                        if x == 0:
                            P_BAL_inflexible_hourly[hrs_year,y,HPP] = (np.mean(c_multiplier_BAL[:,HPP])*c_solar_relative[HPP]*CF_solar_hourly[hrs_year,y,HPP] + np.mean(c_multiplier_BAL[:,HPP])*c_wind_relative[HPP]*CF_wind_hourly[hrs_year,y,HPP] + Q_stable_ratio*np.nanmean(P_CONV_hydro_stable_hourly[:,:,HPP]))
                        elif x > 0:
                            # use estimate P_stable,BAL from previous iteration
                            P_BAL_inflexible_hourly[hrs_year,y,HPP] = (np.mean(c_multiplier_BAL[:,HPP])*c_solar_relative[HPP]*CF_solar_hourly[hrs_year,y,HPP] + np.mean(c_multiplier_BAL[:,HPP])*c_wind_relative[HPP]*CF_wind_hourly[hrs_year,y,HPP] + P_BAL_hydro_stable_hourly[hrs_year,y,HPP])
                        
                    # [calculate] P_load according to constraints on overproduction (eq. S11)
                    P_load_BAL = np.nanpercentile(P_BAL_inflexible_hourly[:,:,HPP],f_size)
                    
                    # [loop] across all simulation years to perform optimization
                    for y in range(len(simulation_years)):
                        
                        # [read] vector with hours in each year
                        hrs_year = range(int(hrs_byyear[y]))
                        
                        # [calculate] hourly load time series in MW (eq. S10)
                        L_BAL_hourly[hrs_year,y,HPP] = P_load_BAL*L_norm[hrs_year,y,HPP]
                        
                        # [calculate] load difference P_d (eq. S9)
                        P_BAL_difference_hourly[hrs_year,y,HPP] = P_BAL_inflexible_hourly[hrs_year,y,HPP] - L_BAL_hourly[hrs_year,y,HPP]
                        
                        # [initialize] initial values of volume (m^3), area (m^2), hydraulic head (m) and ramp restrictions (MW/hr) for each simulation year
                        if y == 0:
                            
                            V_BAL_hourly[0,y,HPP] = V_CONV_hourly[0,y,HPP]
                            A_BAL_hourly[0,y,HPP] = A_CONV_hourly[0,y,HPP]
                            h_BAL_hourly[0,y,HPP] = h_CONV_hourly[0,y,HPP]
                            
                            # [calculate] ramping constraint (eq. S16)
                            temp_sgn_turb = 1
                            P_BAL_ramp_restr_hourly[0,y,HPP] = P_r_turb[HPP]*dP_ramp_turb*mins_hr
                                                        
                        else:
                            
                            temp = V_BAL_hourly[:,y-1,HPP]
                            temp = temp[np.isfinite(temp)]
                            V_BAL_hourly[0,y,HPP] = temp[-1]
            
                            temp = A_BAL_hourly[:,y-1,HPP]
                            temp = temp[np.isfinite(temp)]
                            A_BAL_hourly[0,y,HPP] = temp[-1]
            
                            temp = h_BAL_hourly[:,y-1,HPP]
                            temp = temp[np.isfinite(temp)]
                            h_BAL_hourly[0,y,HPP] = temp[-1]
                            
                            
                            # [calculate] ramping constraint (eq. S16)
                            temp = P_BAL_hydro_flexible_hourly[:,y-1,HPP]
                            temp = temp[np.isfinite(temp)]
                            temp_P_difference = P_BAL_difference_hourly[:,y-1,HPP]
                            temp_P_difference = temp_P_difference[np.isfinite(temp_P_difference)]
                            # [calculate] whether ramping up (temp_sgn = 1) or down (temp_sgn = -1)
                            if P_BAL_difference_hourly[0,y,HPP] - temp_P_difference[-1] < 0:
                                temp_sgn_turb = 1
                            else:
                                temp_sgn_turb = -1
                            P_BAL_ramp_restr_hourly[0,y,HPP] = temp[-1] + temp_sgn_turb*P_r_turb[HPP]*dP_ramp_turb*mins_hr
                            if P_BAL_ramp_restr_hourly[0,y,HPP] < 0:
                                P_BAL_ramp_restr_hourly[0,y,HPP] = 0
                            
                        # [loop] over all time steps in each simulation year to calculate reservoir dynamics and hydropower generation
                        for n in hrs_year:
                            
                            # [check] stable outflow is reduced to zero in case of droughts
                            Q_BAL_stable_hourly[n,y,HPP] = Q_BAL_stable_hourly[n,y,HPP] * hydro_BAL_curtailment_factor_hourly[n,y,HPP]
                            
                            # [calculate] flexible hydropower generation in MW (eq. S16 & S17)
                            if P_BAL_difference_hourly[n,y,HPP] < 0:
                                Q_BAL_pot_turb_flexible[n,y,HPP] = np.max([0, Q_max_turb[HPP] - Q_BAL_stable_hourly[n,y,HPP]]) * hydro_BAL_curtailment_factor_hourly[n,y,HPP]
                                # [calculate] if ramping up
                                if temp_sgn_turb == 1:
                                    P_BAL_hydro_flexible_hourly[n,y,HPP] = np.min([Q_BAL_pot_turb_flexible[n,y,HPP]*eta_turb*rho*g*h_BAL_hourly[n,y,HPP]/10**6, np.min([np.abs(P_BAL_difference_hourly[n,y,HPP]), P_BAL_ramp_restr_hourly[n,y,HPP]]) ])
                                # [calculate] if ramping down
                                elif temp_sgn_turb == -1:
                                    P_BAL_hydro_flexible_hourly[n,y,HPP] = np.min([Q_BAL_pot_turb_flexible[n,y,HPP]*eta_turb*rho*g*h_BAL_hourly[n,y,HPP]/10**6, np.max([np.abs(P_BAL_difference_hourly[n,y,HPP]), P_BAL_ramp_restr_hourly[n,y,HPP]]) ])
                                
                            # [check] flexible hydropower generation is zero when P_d >= 0 (eq. S16)
                            if P_BAL_difference_hourly[n,y,HPP] >= 0:
                                P_BAL_hydro_flexible_hourly[n,y,HPP] = 0
                                
                            # [calculate] stable hydropower generation in MW (eq. S15)
                            Q_pot_turb_BAL = np.min([Q_BAL_stable_hourly[n,y,HPP], Q_max_turb[HPP]])
                            P_BAL_hydro_stable_hourly[n,y,HPP] = Q_pot_turb_BAL*eta_turb*rho*g*h_BAL_hourly[n,y,HPP]/10**6
                                                            
                            # [calculate] flexible turbined flow in m^3/s (eq. S18)
                            if h_BAL_hourly[n,y,HPP] > 0:
                                Q_BAL_flexible_hourly[n,y,HPP] = P_BAL_hydro_flexible_hourly[n,y,HPP]/(eta_turb*rho*g*h_BAL_hourly[n,y,HPP])*10**6
                            else:
                                # [check] cannot be negative
                                h_BAL_hourly[n,y,HPP] = 0
                                Q_BAL_flexible_hourly[n,y,HPP] = 0
                            
                            # [calculate] hydropower generation from RoR flow component in MW (eq. S32)
                            P_BAL_hydro_RoR_hourly[n,y,HPP] = np.min([Q_in_RoR_hourly[n,y,HPP], np.max([0, Q_max_turb[HPP] - Q_BAL_stable_hourly[n,y,HPP] - Q_BAL_flexible_hourly[n,y,HPP]]) ])*eta_turb*rho*g*h_CONV_hourly[n,y,HPP]/10**6
                            
                            # [calculate] spilling component in m^3/s (eq. S19)
                            if V_BAL_hourly[n,y,HPP]/V_max[HPP] < f_spill:
                                Q_BAL_spill_hourly[n,y,HPP] = 0
                            else:
                                Q_BAL_spill_hourly[n,y,HPP] = (Q_in_frac_hourly[n,y,HPP] + (precipitation_flux_hourly[n,y,HPP] - evaporation_flux_hourly[n,y,HPP])*A_BAL_hourly[n,y,HPP]/rho)*(1 + mu) - Q_BAL_stable_hourly[n,y,HPP] - Q_BAL_flexible_hourly[n,y,HPP]
                            # [check] spilling component cannot be negative (eq. S7)
                            if Q_BAL_spill_hourly[n,y,HPP] < 0:
                                Q_BAL_spill_hourly[n,y,HPP] = 0
                            
                            # [calculate] total net outflow in m^3/s (eq. S2)
                            Q_BAL_out_hourly[n,y,HPP] = Q_BAL_stable_hourly[n,y,HPP] + Q_BAL_flexible_hourly[n,y,HPP] + Q_BAL_spill_hourly[n,y,HPP] + Q_in_RoR_hourly[n,y,HPP]

                            # [calculate] reservoir volume in m^3 at next time step (eq. S3, S31)
                            V_BAL_hourly[n+1,y,HPP] = V_BAL_hourly[n,y,HPP] + (Q_in_frac_hourly[n,y,HPP] - Q_BAL_stable_hourly[n,y,HPP] - Q_BAL_flexible_hourly[n,y,HPP] - Q_BAL_spill_hourly[n,y,HPP] + (precipitation_flux_hourly[n,y,HPP] - evaporation_flux_hourly[n,y,HPP])*A_BAL_hourly[n,y,HPP]/rho)*secs_hr
                            
                            # [check] prevent unreal values when lake levels drop low
                            if V_BAL_hourly[n+1,y,HPP] < 0:
                                Q_BAL_stable_hourly[n,y,HPP] = 0
                                P_BAL_hydro_stable_hourly[n,y,HPP] = 0
                                Q_BAL_flexible_hourly[n,y,HPP] = 0
                                P_BAL_hydro_flexible_hourly[n,y,HPP] = 0
                                Q_BAL_out_hourly[n,y,HPP] = Q_BAL_stable_hourly[n,y,HPP] + Q_BAL_flexible_hourly[n,y,HPP] + Q_BAL_spill_hourly[n,y,HPP] + Q_in_RoR_hourly[n,y,HPP]
                                A_BAL_hourly[n,y,HPP] = 0
                                V_BAL_hourly[n+1,y,HPP] = V_BAL_hourly[n,y,HPP] + (Q_in_frac_hourly[n,y,HPP] - Q_BAL_stable_hourly[n,y,HPP] - Q_BAL_flexible_hourly[n,y,HPP] - Q_BAL_spill_hourly[n,y,HPP] + (precipitation_flux_hourly[n,y,HPP] - evaporation_flux_hourly[n,y,HPP])*A_BAL_hourly[n,y,HPP]/rho)*secs_hr
                                
                            # [calculate] reservoir lake area in m^2 and hydraulic head in m from bathymetric relationship
                            h_temp = np.where(abs(calibrate_volume[:,HPP] - V_BAL_hourly[n+1,y,HPP]) == min(abs(calibrate_volume[:,HPP] - V_BAL_hourly[n+1,y,HPP])))[0][0]
                            A_BAL_hourly[n+1,y,HPP] = calibrate_area[h_temp,HPP]
                            h_BAL_hourly[n+1,y,HPP] = calibrate_head[h_temp,HPP]
                            
                            # [calculate] ramp rate restrictions (MW attainable) at next time step (eq. S16)
                            if n < len(hrs_year) - 1:
                                if (P_BAL_difference_hourly[n+1,y,HPP] - P_BAL_difference_hourly[n,y,HPP]) < 0:
                                    temp_sgn_turb = 1
                                else:
                                    temp_sgn_turb = -1
                                P_BAL_ramp_restr_hourly[n+1,y,HPP] = P_BAL_hydro_flexible_hourly[n,y,HPP] + temp_sgn_turb*P_r_turb[HPP]*dP_ramp_turb*mins_hr
                                if P_BAL_ramp_restr_hourly[n+1,y,HPP] < 0:
                                    P_BAL_ramp_restr_hourly[n+1,y,HPP] = 0
                                
                            # [calculate] whether lake levels have dropped so low as to require hydropower curtailment
                            # [calculate] for small HPPs: use "RoR" flow component to fill up reservoir in case water levels have dropped below f_release*V_max
                            # (see explanation below eq. S33)
                            if HPP_category[HPP] == "B":
                                if V_BAL_hourly[n+1,y,HPP] < f_restart*V_max[HPP]:
                                    if n < len(hrs_year) - 1:
                                        Q_in_frac_hourly[n+1,y,HPP] = Q_in_frac_hourly[n+1,y,HPP] + Q_in_RoR_hourly[n+1,y,HPP]
                                        Q_in_RoR_hourly[n+1,y,HPP] = 0
                                    elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                                        Q_in_frac_hourly[0,y+1,HPP] = Q_in_frac_hourly[0,y+1,HPP] + Q_in_RoR_hourly[0,y+1,HPP]
                                        Q_in_RoR_hourly[0,y+1,HPP] = 0
                            
                            # [calculate] for large and small HPPs: curtail hydropower generation in case water levels have dropped below f_stop*V_max
                            # (see section S3.1)
                            if V_BAL_hourly[n+1,y,HPP] < f_stop*V_max[HPP]:
                                if n < len(hrs_year) - 1:
                                    hydro_BAL_curtailment_factor_hourly[n+1,y,HPP] = 0
                                elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                                    hydro_BAL_curtailment_factor_hourly[0,y+1,HPP] = 0
                                    
                            # [calculate] restart hydropower generation if reservoir levels have recovered
                            # (see section S3.1)
                            if hydro_BAL_curtailment_factor_hourly[n,y,HPP] == 0 and V_BAL_hourly[n+1,y,HPP] > f_restart*V_max[HPP]:
                                if n < len(hrs_year) - 1:
                                    hydro_BAL_curtailment_factor_hourly[n+1,y,HPP] = 1
                                elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                                    hydro_BAL_curtailment_factor_hourly[0,y+1,HPP] = 1
                            elif hydro_BAL_curtailment_factor_hourly[n,y,HPP] == 0 and V_BAL_hourly[n+1,y,HPP] <= f_restart*V_max[HPP]:
                                if n < len(hrs_year) - 1:
                                    hydro_BAL_curtailment_factor_hourly[n+1,y,HPP] = 0
                                elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                                    hydro_BAL_curtailment_factor_hourly[0,y+1,HPP] = 0
                            
                        
                    
                # [arrange] complete time series of water volume for eq. S20
                for y in range(len(simulation_years)):
                    V_BAL_hourly[int(hrs_byyear[y]),y,HPP] = np.nan
                
                temp_volume_upper_BAL_series = V_BAL_hourly[:,:,HPP]
                temp_volume_upper_BAL_series = (np.transpose(temp_volume_upper_BAL_series)).ravel()
                temp_volume_upper_BAL_series = temp_volume_upper_BAL_series[np.isfinite(temp_volume_upper_BAL_series)]
                
                # [calculate] deviation between CONV and BAL reservoir dynamics (eq. S21)
                psi_BAL[f] = np.mean(np.abs(temp_volume_upper_BAL_series - V_CONV_series_hourly[:,HPP]))/np.mean(V_CONV_series_hourly[:,HPP])
                
                # [check] see explanation below eq. S21: if droughts occur in CONV, BAL should have no MORE days of curtailed flow than CONV ...
                # and curtailed flow should occur in less than 50% of the years in the simulation, so median yearly statistics represent normal operation
                if np.nanmin(V_CONV_hourly[:,:,HPP]) < f_stop*V_max[HPP] and (np.nansum(Q_BAL_out_hourly[:,:,HPP] == 0) > np.nansum(Q_CONV_out_hourly[:,:,HPP] == 0) or np.sum(np.sum(Q_BAL_out_hourly[:,:,HPP] - Q_in_RoR_hourly[:,:,HPP] == 0, axis = 0) > 0) > np.floor(len(simulation_years)/2)):
                    psi_BAL[f] = np.nan
                    # [check] if droughts do not occur in CONV, then neither should they in BAL
                elif np.nanmin(V_CONV_hourly[:,:,HPP]) >= f_stop*V_max[HPP] and np.nanmin(V_BAL_hourly[:,:,HPP]) < f_stop*V_max[HPP]:
                    psi_BAL[f] = np.nan
                
            # psi_BAL
            
            # [identify] minimum in psi (eq. S21)
            if np.sum(np.isnan(psi_BAL)) == len(psi_BAL) and f_demand_BAL[0] == 0:
                f_demand_opt_BAL = 0
                psi_BAL_opt = 0
                break
            else:
                crossing_BAL = np.where(psi_BAL == min(psi_BAL))[0][0]
                f_demand_opt_BAL = f_demand_BAL[crossing_BAL]
                psi_BAL_opt = np.abs(psi_BAL[crossing_BAL])
            
            # [check] prevent negative results
            if f_demand_opt_BAL == 0:
                f_demand_opt_BAL = f_demand_BAL[crossing_BAL + 1]
                psi_BAL_opt = np.abs(psi_BAL[crossing_BAL + 1])
                
            # [check] determine if psi is low enough for this to qualify as optimum solution
            if psi_BAL_opt < psi_min_threshold:
                break
            
            # [check] if range in which to identify ELCC is adequate
            if f_demand_opt_BAL == f_demand_BAL[-1]:
                print("Warning: parameter f_init_BAL_end likely set too low")
            
        # [initialize] optimal value of total SW power (MWh/year) so that we can calculate optimal c_solar and c_wind (eq. S25)
        E_SW_loop_BAL_opt[HPP] = np.mean(E_hydro_CONV_stable_yearly[:,HPP])*f_demand_opt_BAL
        
        
        ###############################################################
        ############------------ BAL optimized ------------############
        ###############################################################
        
        # [display]
        print("(iii) found optimum BAL solution - saving all variables")
        
        # [preallocate] to test convergence towards P_stable (see explanation below eq. S19)
        convergence_test_BAL = np.zeros(shape = (X_max_BAL))
        
        # [loop] across all simulation years to identify realistic c_solar and c_wind values
        for y in range(len(simulation_years)):
            
            # [read] vector with hours in each year
            hrs_year = range(int(hrs_byyear[y]))
            
            # [calculate] determine realistic amount of SW capacity in MW (c_solar, c_wind) corresponding to generation equal to E_SW_loop_BAL
            E_SW_per_MW_BAL_yearly[y,HPP] = np.sum(c_solar_relative[HPP]*CF_solar_hourly[hrs_year,y,HPP] + c_wind_relative[HPP]*CF_wind_hourly[hrs_year,y,HPP])
            c_multiplier_BAL[y,HPP] = E_SW_loop_BAL_opt[HPP]/E_SW_per_MW_BAL_yearly[y,HPP]
        
        # [loop] perform iterations to get converged estimate of P_stable (see explanation below eq. S19)
        for x in range(X_max_BAL):
                                    
            # [calculate] environmentally required outflow (eq. S14)
            temp_Q_out_BAL = Q_in_nat_av*np.ones(shape = (len(Q_CONV_stable_hourly),len(Q_CONV_stable_hourly[0])))
            temp_Q_out_BAL[np.isnan(Q_CONV_stable_hourly[:,:,HPP])] = np.nan
            Q_BAL_stable_hourly[:,:,HPP] = Q_stable_ratio*temp_Q_out_BAL
                    
            # [initialize] ensure Q_in_frac_hourly and Q_in_RoR_hourly are written correctly at the beginning of each step in the loop
            Q_in_frac_hourly[:,:,HPP] = Q_in_frac_store[:,:,HPP]
            Q_in_RoR_hourly[:,:,HPP] = Q_in_RoR_store[:,:,HPP]
                    
            # [initialize] This variable is equal to unity by default, but set to zero in case of extreme droughts forcing a
            # temporary curtailment on hydropower generation (section S3.1)
            hydro_BAL_curtailment_factor_hourly[:,:,HPP] = 1
            
            # [loop] across all simulation years to initialize P_stable (see explanation below eq. S19)
            for y in range(len(simulation_years)):
                    
                # [read] vector with hours in each year
                hrs_year = range(int(hrs_byyear[y]))

                # [initialize] stable hydropower generation P_stable in MW (see explanation below eq. S19)
                # use estimate P_stable,BAL = (1 - C_OR)*P_stable,CONV as initial guess
                if x == 0:
                    P_BAL_inflexible_hourly[hrs_year,y,HPP] = (np.mean(c_multiplier_BAL[:,HPP])*c_solar_relative[HPP]*CF_solar_hourly[hrs_year,y,HPP] + np.mean(c_multiplier_BAL[:,HPP])*c_wind_relative[HPP]*CF_wind_hourly[hrs_year,y,HPP] + Q_stable_ratio*np.nanmean(P_CONV_hydro_stable_hourly[:,:,HPP]))
                elif x > 0:
                    # use estimate P_stable,BAL from previous iteration
                    P_BAL_inflexible_hourly[hrs_year,y,HPP] = (np.mean(c_multiplier_BAL[:,HPP])*c_solar_relative[HPP]*CF_solar_hourly[hrs_year,y,HPP] + np.mean(c_multiplier_BAL[:,HPP])*c_wind_relative[HPP]*CF_wind_hourly[hrs_year,y,HPP] + P_BAL_hydro_stable_hourly[hrs_year,y,HPP])
                    
                # [calculate] total solar and wind power generation by hour (eq. S12)
                P_BAL_solar_hourly[hrs_year,y,HPP] = np.mean(c_multiplier_BAL[:,HPP])*c_solar_relative[HPP]*CF_solar_hourly[hrs_year,y,HPP]
                P_BAL_wind_hourly[hrs_year,y,HPP] = np.mean(c_multiplier_BAL[:,HPP])*c_wind_relative[HPP]*CF_wind_hourly[hrs_year,y,HPP]
                
            
            # [calculate] P_load according to constraints on overproduction (eq. S11)
            P_load_BAL = np.nanpercentile(P_BAL_inflexible_hourly[:,:,HPP],f_size)

            # [loop] across all simulation years to perform optimization
            for y in range(len(simulation_years)):
                
                # [read] vector with hours in each year
                hrs_year = range(int(hrs_byyear[y]))
                
                # [calculate] hourly load time series in MW (eq. S10)
                L_BAL_hourly[hrs_year,y,HPP] = P_load_BAL*L_norm[hrs_year,y,HPP]
                
                # [calculate] load difference P_d (eq. S9)
                P_BAL_difference_hourly[hrs_year,y,HPP] = P_BAL_inflexible_hourly[hrs_year,y,HPP] - L_BAL_hourly[hrs_year,y,HPP]
                
                # [initialize] initial values of volume (m^3), area (m^2), hydraulic head (m) and ramp restrictions (MW/hr) for each simulation year
                if y == 0:
                    
                    V_BAL_hourly[0,y,HPP] = V_CONV_hourly[0,y,HPP]
                    A_BAL_hourly[0,y,HPP] = A_CONV_hourly[0,y,HPP]
                    h_BAL_hourly[0,y,HPP] = h_CONV_hourly[0,y,HPP]
                    
                    # [calculate] ramping constraint (eq. S16)
                    temp_sgn_turb = 1
                    P_BAL_ramp_restr_hourly[0,y,HPP] = P_r_turb[HPP]*dP_ramp_turb*mins_hr
                                                
                else:
                    
                    temp = V_BAL_hourly[:,y-1,HPP]
                    temp = temp[np.isfinite(temp)]
                    V_BAL_hourly[0,y,HPP] = temp[-1]
    
                    temp = A_BAL_hourly[:,y-1,HPP]
                    temp = temp[np.isfinite(temp)]
                    A_BAL_hourly[0,y,HPP] = temp[-1]
    
                    temp = h_BAL_hourly[:,y-1,HPP]
                    temp = temp[np.isfinite(temp)]
                    h_BAL_hourly[0,y,HPP] = temp[-1]
                    
                    
                    # [calculate] ramping constraint (eq. S16)
                    temp = P_BAL_hydro_flexible_hourly[:,y-1,HPP]
                    temp = temp[np.isfinite(temp)]
                    temp_P_difference = P_BAL_difference_hourly[:,y-1,HPP]
                    temp_P_difference = temp_P_difference[np.isfinite(temp_P_difference)]
                    # [calculate] whether ramping up (temp_sgn = 1) or down (temp_sgn = -1)
                    if P_BAL_difference_hourly[0,y,HPP] - temp_P_difference[-1] < 0:
                        temp_sgn_turb = 1
                    else:
                        temp_sgn_turb = -1
                    P_BAL_ramp_restr_hourly[0,y,HPP] = temp[-1] + temp_sgn_turb*P_r_turb[HPP]*dP_ramp_turb*mins_hr
                    if P_BAL_ramp_restr_hourly[0,y,HPP] < 0:
                        P_BAL_ramp_restr_hourly[0,y,HPP] = 0
                    
                # [loop] over all time steps in each simulation year to calculate reservoir dynamics and hydropower generation
                for n in hrs_year:
                    
                    # [check] stable outflow is reduced to zero in case of droughts
                    Q_BAL_stable_hourly[n,y,HPP] = Q_BAL_stable_hourly[n,y,HPP] * hydro_BAL_curtailment_factor_hourly[n,y,HPP]
                    
                    # [calculate] flexible hydropower generation in MW (eq. S16 & S17)
                    if P_BAL_difference_hourly[n,y,HPP] < 0:
                        Q_BAL_pot_turb_flexible[n,y,HPP] = np.max([0, Q_max_turb[HPP] - Q_BAL_stable_hourly[n,y,HPP]]) * hydro_BAL_curtailment_factor_hourly[n,y,HPP]
                        # [calculate] if ramping up
                        if temp_sgn_turb == 1:
                            P_BAL_hydro_flexible_hourly[n,y,HPP] = np.min([Q_BAL_pot_turb_flexible[n,y,HPP]*eta_turb*rho*g*h_BAL_hourly[n,y,HPP]/10**6, np.min([np.abs(P_BAL_difference_hourly[n,y,HPP]), P_BAL_ramp_restr_hourly[n,y,HPP]]) ])
                        # [calculate] if ramping down
                        elif temp_sgn_turb == -1:
                            P_BAL_hydro_flexible_hourly[n,y,HPP] = np.min([Q_BAL_pot_turb_flexible[n,y,HPP]*eta_turb*rho*g*h_BAL_hourly[n,y,HPP]/10**6, np.max([np.abs(P_BAL_difference_hourly[n,y,HPP]), P_BAL_ramp_restr_hourly[n,y,HPP]]) ])
                        
                    # [check] flexible hydropower generation is zero when P_d >= 0 (eq. S16)
                    if P_BAL_difference_hourly[n,y,HPP] >= 0:
                        P_BAL_hydro_flexible_hourly[n,y,HPP] = 0
                        
                    # [calculate] stable hydropower generation in MW (eq. S15)
                    Q_pot_turb_BAL = np.min([Q_BAL_stable_hourly[n,y,HPP], Q_max_turb[HPP]])
                    P_BAL_hydro_stable_hourly[n,y,HPP] = Q_pot_turb_BAL*eta_turb*rho*g*h_BAL_hourly[n,y,HPP]/10**6
                    
                    # [calculate] flexible turbined flow in m^3/s (eq. S18)
                    if h_BAL_hourly[n,y,HPP] > 0:
                        Q_BAL_flexible_hourly[n,y,HPP] = P_BAL_hydro_flexible_hourly[n,y,HPP]/(eta_turb*rho*g*h_BAL_hourly[n,y,HPP])*10**6
                    else:
                        # [check] cannot be negative
                        h_BAL_hourly[n,y,HPP] = 0
                        Q_BAL_flexible_hourly[n,y,HPP] = 0
                    
                    # [calculate] hydropower generation from RoR flow component in MW (eq. S32)
                    P_BAL_hydro_RoR_hourly[n,y,HPP] = np.min([Q_in_RoR_hourly[n,y,HPP], np.max([0, Q_max_turb[HPP] - Q_BAL_stable_hourly[n,y,HPP] - Q_BAL_flexible_hourly[n,y,HPP]]) ])*eta_turb*rho*g*h_CONV_hourly[n,y,HPP]/10**6
                    
                    # [calculate] spilling component in m^3/s (eq. S19)
                    if V_BAL_hourly[n,y,HPP]/V_max[HPP] < f_spill:
                        Q_BAL_spill_hourly[n,y,HPP] = 0
                    else:
                        Q_BAL_spill_hourly[n,y,HPP] = (Q_in_frac_hourly[n,y,HPP] + (precipitation_flux_hourly[n,y,HPP] - evaporation_flux_hourly[n,y,HPP])*A_BAL_hourly[n,y,HPP]/rho)*(1 + mu) - Q_BAL_stable_hourly[n,y,HPP] - Q_BAL_flexible_hourly[n,y,HPP]
                    # [check] spilling component cannot be negative (eq. S7)
                    if Q_BAL_spill_hourly[n,y,HPP] < 0:
                        Q_BAL_spill_hourly[n,y,HPP] = 0
                    
                    # [calculate] total net outflow in m^3/s (eq. S2)
                    Q_BAL_out_hourly[n,y,HPP] = Q_BAL_stable_hourly[n,y,HPP] + Q_BAL_flexible_hourly[n,y,HPP] + Q_BAL_spill_hourly[n,y,HPP] + Q_in_RoR_hourly[n,y,HPP]

                    # [calculate] reservoir volume in m^3 at next time step (eq. S3, S31)
                    V_BAL_hourly[n+1,y,HPP] = V_BAL_hourly[n,y,HPP] + (Q_in_frac_hourly[n,y,HPP] - Q_BAL_stable_hourly[n,y,HPP] - Q_BAL_flexible_hourly[n,y,HPP] - Q_BAL_spill_hourly[n,y,HPP] + (precipitation_flux_hourly[n,y,HPP] - evaporation_flux_hourly[n,y,HPP])*A_BAL_hourly[n,y,HPP]/rho)*secs_hr
                    
                    # [check] prevent unreal values when lake levels drop low
                    if V_BAL_hourly[n+1,y,HPP] < 0:
                        Q_BAL_stable_hourly[n,y,HPP] = 0
                        P_BAL_hydro_stable_hourly[n,y,HPP] = 0
                        Q_BAL_flexible_hourly[n,y,HPP] = 0
                        P_BAL_hydro_flexible_hourly[n,y,HPP] = 0
                        Q_BAL_out_hourly[n,y,HPP] = Q_BAL_stable_hourly[n,y,HPP] + Q_BAL_flexible_hourly[n,y,HPP] + Q_BAL_spill_hourly[n,y,HPP] + Q_in_RoR_hourly[n,y,HPP]
                        A_BAL_hourly[n,y,HPP] = 0
                        V_BAL_hourly[n+1,y,HPP] = V_BAL_hourly[n,y,HPP] + (Q_in_frac_hourly[n,y,HPP] - Q_BAL_stable_hourly[n,y,HPP] - Q_BAL_flexible_hourly[n,y,HPP] - Q_BAL_spill_hourly[n,y,HPP] + (precipitation_flux_hourly[n,y,HPP] - evaporation_flux_hourly[n,y,HPP])*A_BAL_hourly[n,y,HPP]/rho)*secs_hr
                        
                    # [calculate] reservoir lake area in m^2 and hydraulic head in m from bathymetric relationship
                    h_temp = np.where(abs(calibrate_volume[:,HPP] - V_BAL_hourly[n+1,y,HPP]) == min(abs(calibrate_volume[:,HPP] - V_BAL_hourly[n+1,y,HPP])))[0][0]
                    A_BAL_hourly[n+1,y,HPP] = calibrate_area[h_temp,HPP]
                    h_BAL_hourly[n+1,y,HPP] = calibrate_head[h_temp,HPP]
                    
                    # [calculate] ramp rate restrictions (MW attainable) at next time step (eq. S16)
                    if n < len(hrs_year) - 1:
                        if (P_BAL_difference_hourly[n+1,y,HPP] - P_BAL_difference_hourly[n,y,HPP]) < 0:
                            temp_sgn_turb = 1
                        else:
                            temp_sgn_turb = -1
                        P_BAL_ramp_restr_hourly[n+1,y,HPP] = P_BAL_hydro_flexible_hourly[n,y,HPP] + temp_sgn_turb*P_r_turb[HPP]*dP_ramp_turb*mins_hr
                        if P_BAL_ramp_restr_hourly[n+1,y,HPP] < 0:
                            P_BAL_ramp_restr_hourly[n+1,y,HPP] = 0
                        
                    # [calculate] whether lake levels have dropped so low as to require hydropower curtailment
                    # [calculate] for small HPPs: use "RoR" flow component to fill up reservoir in case water levels have dropped below f_release*V_max
                    # (see explanation below eq. S33)
                    if HPP_category[HPP] == "B":
                        if V_BAL_hourly[n+1,y,HPP] < f_restart*V_max[HPP]:
                            if n < len(hrs_year) - 1:
                                Q_in_frac_hourly[n+1,y,HPP] = Q_in_frac_hourly[n+1,y,HPP] + Q_in_RoR_hourly[n+1,y,HPP]
                                Q_in_RoR_hourly[n+1,y,HPP] = 0
                            elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                                Q_in_frac_hourly[0,y+1,HPP] = Q_in_frac_hourly[0,y+1,HPP] + Q_in_RoR_hourly[0,y+1,HPP]
                                Q_in_RoR_hourly[0,y+1,HPP] = 0
                    
                    # [calculate] for large and small HPPs: curtail hydropower generation in case water levels have dropped below f_stop*V_max
                    # (see section S3.1)
                    if V_BAL_hourly[n+1,y,HPP] < f_stop*V_max[HPP]:
                        if n < len(hrs_year) - 1:
                            hydro_BAL_curtailment_factor_hourly[n+1,y,HPP] = 0
                        elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                            hydro_BAL_curtailment_factor_hourly[0,y+1,HPP] = 0
                            
                    # [calculate] restart hydropower generation if reservoir levels have recovered
                    # (see section S3.1)
                    if hydro_BAL_curtailment_factor_hourly[n,y,HPP] == 0 and V_BAL_hourly[n+1,y,HPP] > f_restart*V_max[HPP]:
                        if n < len(hrs_year) - 1:
                            hydro_BAL_curtailment_factor_hourly[n+1,y,HPP] = 1
                        elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                            hydro_BAL_curtailment_factor_hourly[0,y+1,HPP] = 1
                    elif hydro_BAL_curtailment_factor_hourly[n,y,HPP] == 0 and V_BAL_hourly[n+1,y,HPP] <= f_restart*V_max[HPP]:
                        if n < len(hrs_year) - 1:
                            hydro_BAL_curtailment_factor_hourly[n+1,y,HPP] = 0
                        elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                            hydro_BAL_curtailment_factor_hourly[0,y+1,HPP] = 0
                            
                    
                # [calculate] total solar and wind power generation under optimal BAL solution in MWh/year (eq. S25)
                E_solar_BAL_yearly[y,HPP] = np.sum(P_BAL_solar_hourly[hrs_year,y,HPP])
                E_wind_BAL_yearly[y,HPP] = np.sum(P_BAL_wind_hourly[hrs_year,y,HPP])
                E_SW_BAL_yearly[y,HPP] = E_solar_BAL_yearly[y,HPP] + E_wind_BAL_yearly[y,HPP]
                
                # [calculate] share of solar (= 1 - share of wind) in the resulting SW mix
                share_SW_BAL[y,HPP] = E_solar_BAL_yearly[y,HPP]/(E_solar_BAL_yearly[y,HPP] + E_wind_BAL_yearly[y,HPP])
                
                # [calculate] total flexible hydropower generation under optimal BAL solution in MWh/year (eq. S24)
                E_hydro_BAL_flexible_yearly[y,HPP] = np.sum(P_BAL_hydro_flexible_hourly[hrs_year,y,HPP])
                
                # [calculate] total stable hydropower generation under optimal BAL solution in MWh/year (eq. S24)
                E_hydro_BAL_stable_yearly[y,HPP] = np.sum(P_BAL_hydro_stable_hourly[hrs_year,y,HPP])
                
                # [calculate] total stable + flexible hydropower generation under optimal BAL solution in MWh/year (eq. S24)
                E_hydro_BAL_nonRoR_yearly[y,HPP] = E_hydro_BAL_flexible_yearly[y,HPP] + E_hydro_BAL_stable_yearly[y,HPP]
                
                # [calculate] total RoR hydropower generation under optimal BAL solution in MWh/year (eq. S33)
                E_hydro_BAL_RoR_yearly[y,HPP] = np.sum(P_BAL_hydro_RoR_hourly[hrs_year,y,HPP])
                
                # [calculate] total HSW generation under optimal BAL solution in MWh/year (excluding RoR component)
                temp_power_delivered_BAL = P_BAL_solar_hourly[hrs_year,y,HPP] + P_BAL_wind_hourly[hrs_year,y,HPP] + P_BAL_hydro_stable_hourly[hrs_year,y,HPP] + P_BAL_hydro_flexible_hourly[hrs_year,y,HPP]
                
                # [calculate] net difference between generation and load under optimal BAL solution in MWh/year
                temp_power_diff_BAL = temp_power_delivered_BAL - L_BAL_hourly[hrs_year,y,HPP]

                # [calculate] yearly overproduction compared to load under optimal BAL solution in MWh/year
                E_overproduced_BAL_yearly[y,HPP] = np.abs(np.sum(temp_power_diff_BAL[temp_power_diff_BAL > 0]))

                # [calculate] yearly SW generation minus overproduction under optimal BAL solution in MWh/year
                E_SW_BAL_without_overproduction[y,HPP] = E_solar_BAL_yearly[y,HPP] + E_wind_BAL_yearly[y,HPP] - E_overproduced_BAL_yearly[y,HPP]
                
                
                ##### IDENTIFY YEARLY ELCC #####

                # [calculate] total supplied HSW generation under optimal BAL solution
                total_power_supply_BAL = P_BAL_hydro_stable_hourly[hrs_year,y,HPP] + P_BAL_hydro_flexible_hourly[hrs_year,y,HPP] + np.mean(c_multiplier_BAL[:,HPP])*c_solar_relative[HPP]*CF_solar_hourly[hrs_year,y,HPP] + np.mean(c_multiplier_BAL[:,HPP])*c_wind_relative[HPP]*CF_wind_hourly[hrs_year,y,HPP]
                N_power_supply_BAL = int(np.ceil(np.max(total_power_supply_BAL)))

                # [preallocate] range in which to identify ELCC
                P_followed_BAL_range[y,:,HPP] = np.linspace(0,N_power_supply_BAL,N_ELCC)
                power_unmet_BAL = np.zeros(shape = N_ELCC)
                
                # [loop] to identify ELCC under optimal BAL solution
                for n in range(N_ELCC):
                    temp = total_power_supply_BAL - P_followed_BAL_range[y,n,HPP]*L_norm[hrs_year,y,HPP]
                    if np.abs(np.mean(temp[temp<=0])) > 0:
                        power_unmet_BAL[n] = np.abs(np.sum(temp[temp<=0]))/np.sum(P_followed_BAL_range[y,n,HPP]*L_norm[hrs_year,y,HPP])
                
                # [identify] total P_followed given the constraint LOEE_allowed (default zero)
                N_demand_covered_BAL_temp = np.where(power_unmet_BAL[power_unmet_BAL != np.Inf] > LOEE_allowed)[0][0] - 1
                if N_demand_covered_BAL_temp.size == 0 or N_demand_covered_BAL_temp == 0:
                    P_followed_BAL_index[y,HPP] = 1
                else:
                    P_followed_BAL_index[y,HPP] = N_demand_covered_BAL_temp
                
                # [identify] hourly time series of L_followed (MW) (eq. S23)
                L_followed_BAL_hourly[hrs_year,y,HPP] = P_followed_BAL_range[y,int(P_followed_BAL_index[y,HPP]),HPP]*L_norm[hrs_year,y,HPP]
                
                # [calculate] ELCC by year (MWh/year) (eq. S23)
                ELCC_BAL_yearly[y,HPP] = np.sum(L_followed_BAL_hourly[hrs_year,y,HPP])

                # [calculate] difference between ELCC and total HSW generated (excl. RoR component) to obtain Residual Load Duration Curve (RLDC) (eq. S22)
                L_res_BAL_hourly[hrs_year,y,HPP] = L_followed_BAL_hourly[hrs_year,y,HPP] - total_power_supply_BAL
                
                # [arrange] mean fraction of unmet load by month
                for m in range(months_yr):
                    temp1 = L_res_BAL_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP]
                    temp2 = L_followed_BAL_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP]
                    L_unmet_BAL_frac_bymonth[m,y,HPP] = np.sum(temp1[temp1>0])/np.sum(temp2)
                    
                # [arrange] yearly average outflow under optimal BAL solution
                Q_BAL_out_yearly[y,HPP] = np.mean(Q_BAL_out_hourly[hrs_year,y,HPP])
                
                
            # [check] to check convergence of solution towards P_stable
            convergence_test_BAL[x] = np.nanmean(P_BAL_hydro_stable_hourly[:,:,HPP])
            
            
        # [arrange] outflow data by month
        for y in range(len(simulation_years)):
            for m in range(months_yr):
                Q_BAL_out_monthly[m,y,HPP] = np.mean(Q_BAL_out_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP])
                Q_used_as_nonRoR_monthly[m,y,HPP] = np.mean(Q_in_frac_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP])
                Q_used_as_RoR_monthly[m,y,HPP] = np.mean(Q_in_RoR_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP])
                
        
        # [arrange] complete time series of water volume, area and levels
        for y in range(len(simulation_years)): 
            V_BAL_hourly[int(hrs_byyear[y]),y,HPP] = np.nan
            A_BAL_hourly[int(hrs_byyear[y]),y,HPP] = np.nan
            h_BAL_hourly[int(hrs_byyear[y]),y,HPP] = np.nan
            
        temp_volume_upper_BAL_series = V_BAL_hourly[:,:,HPP]
        temp_volume_upper_BAL_series = (np.transpose(temp_volume_upper_BAL_series)).ravel()
        temp_volume_upper_BAL_series = temp_volume_upper_BAL_series[np.isfinite(temp_volume_upper_BAL_series)]
        V_BAL_series_hourly[:,HPP] = temp_volume_upper_BAL_series
        
        temp_area_BAL_series = A_BAL_hourly[:,:,HPP]
        temp_area_BAL_series = (np.transpose(temp_area_BAL_series)).ravel()
        temp_area_BAL_series = temp_area_BAL_series[np.isfinite(temp_area_BAL_series)]
        A_BAL_series_hourly[:,HPP] = temp_area_BAL_series
        
        temp_head_BAL_series = h_BAL_hourly[:,:,HPP]
        temp_head_BAL_series = (np.transpose(temp_head_BAL_series)).ravel()
        temp_head_BAL_series = temp_head_BAL_series[np.isfinite(temp_head_BAL_series)]
        h_BAL_series_hourly[:,HPP] = temp_head_BAL_series
        
        # [display] once BAL simulation is complete
        print("done")
        
        
        ###############################################################
        ############----STATISTICS FOR DATA PARSING -------############
        ###############################################################
        
        # [arrange] medians in hydropower generation in GWh/year
        E_hydro_CONV_stable_statistics_median[HPP] = 1e-3*np.median(E_hydro_CONV_stable_yearly[:,HPP])
        E_hydro_CONV_RoR_statistics_median[HPP] = 1e-3*np.median(E_hydro_CONV_RoR_yearly[:,HPP])
        E_hydro_BAL_nonRoR_statistics_median[HPP] = 1e-3*np.median(E_hydro_BAL_nonRoR_yearly[:,HPP])
        E_hydro_BAL_RoR_statistics_median[HPP] = 1e-3*np.median(E_hydro_BAL_RoR_yearly[:,HPP])
        
        # [arrange] yearly totals of HSW generation in MWh/year
        E_HSW_BAL_yearly[:,HPP] = E_hydro_BAL_flexible_yearly[:,HPP] + E_hydro_BAL_stable_yearly[:,HPP] + E_solar_BAL_yearly[:,HPP] + E_wind_BAL_yearly[:,HPP]
        
        # [arrange] medians and IQ ranges of HSW generation and ELCC in GWh/year
        E_HSW_BAL_statistics_median[HPP] = 1e-3*np.median(E_HSW_BAL_yearly[:,HPP])
        E_HSW_BAL_statistics_pct25[HPP] = 1e-3*np.percentile(E_HSW_BAL_yearly[:,HPP],25)
        E_HSW_BAL_statistics_pct75[HPP] = 1e-3*np.percentile(E_HSW_BAL_yearly[:,HPP],75)
        
        E_solar_BAL_statistics_median[HPP] = 1e-3*np.median(E_solar_BAL_yearly[:,HPP])
        E_solar_BAL_statistics_pct25[HPP] = 1e-3*np.percentile(E_solar_BAL_yearly[:,HPP],25)
        E_solar_BAL_statistics_pct75[HPP] = 1e-3*np.percentile(E_solar_BAL_yearly[:,HPP],75)
        
        E_wind_BAL_statistics_median[HPP] = 1e-3*np.median(E_wind_BAL_yearly[:,HPP])
        E_wind_BAL_statistics_pct25[HPP] = 1e-3*np.percentile(E_solar_BAL_yearly[:,HPP],25)
        E_wind_BAL_statistics_pct75[HPP] = 1e-3*np.percentile(E_solar_BAL_yearly[:,HPP],75)
        
        ELCC_BAL_statistics_median[HPP] = 1e-3*np.median(ELCC_BAL_yearly[:,HPP])
        ELCC_BAL_statistics_pct25[HPP] = 1e-3*np.percentile(ELCC_BAL_yearly[:,HPP],25)
        ELCC_BAL_statistics_pct75[HPP] = 1e-3*np.percentile(ELCC_BAL_yearly[:,HPP],75)
        
        # [arrange] ratio of yearly ELCC to yearly hydropower generation (BAL)
        ratio_ELCC_E_hydro_BAL_yearly[:,HPP] = ELCC_BAL_yearly[:,HPP]/(E_hydro_BAL_nonRoR_yearly[:,HPP] + E_hydro_BAL_RoR_yearly[:,HPP])
        ratio_ELCC_E_hydro_BAL_median[HPP] = np.nanmedian(ratio_ELCC_E_hydro_BAL_yearly[:,HPP])
        
        # [calculate] yearly hydropower capacity factor for CONV
        CF_hydro_CONV_yearly[:,HPP] = (E_hydro_CONV_stable_yearly[:,HPP] + E_hydro_CONV_RoR_yearly[:,HPP])/((P_r_turb[HPP])*hrs_byyear)
        
        # [calculate] hourly hydropower capacity factor for BAL (eq. S42)
        CF_hydro_BAL_hourly[:,:,HPP] = (P_BAL_hydro_stable_hourly[:,:,HPP] + P_BAL_hydro_flexible_hourly[:,:,HPP] + P_BAL_hydro_RoR_hourly[:,:,HPP])/(P_r_turb[HPP])
        
        # [calculate] turbine exhaustion factor k_turb in BAL (eq. S28)
        k_turb_hourly_BAL[:,:,HPP] = (Q_BAL_stable_hourly[:,:,HPP] + Q_BAL_flexible_hourly[:,:,HPP])/Q_max_turb[HPP]
        
        # [check] if criterion on k_turb is met for BAL, wrap up simulation and write data
        if np.median(np.nanpercentile(k_turb_hourly_BAL[:,:,HPP],99,0)) < 1:
            # [arrange] data for tables in SI B (column 7-12 in Table S3-7)
            data_SI_B_BAL[HPP,:] = np.array([P_r_turb[HPP], C_OR_range_BAL[q], E_hydro_BAL_nonRoR_statistics_median[HPP], E_hydro_BAL_RoR_statistics_median[HPP], E_solar_BAL_statistics_median[HPP], E_wind_BAL_statistics_median[HPP], ELCC_BAL_statistics_median[HPP]])
            break
        else:
            # [display] in case k_turb criterion was not met (eq. S28)
            print("requires resimulating at lower C_OR...")
    
    
    
    ###############################################################
    ############----------- STOR iterations -----------############
    ###############################################################
        
    # [initialize] STOR scenario is only relevant for large HPPs
    if HPP_category[HPP] == "B":
        STOR_break[HPP] = 1
    elif option_storage == 0:
        STOR_break[HPP] = 1
    
    # [check] start loop if STOR scenario could be an option (0 = yes, 1 = no)
    if STOR_break[HPP] == 0:
        
        # [display] start of iterations to find optimal solution for STOR operation
        print("(iv) finding optimal STOR solution")
        
        for q in range(len(C_OR_range_STOR)):
        
            # [calculate] ratio of stable (environmental) to average total outflow (see eq. S14)
            Q_stable_ratio = 1 - C_OR_range_STOR[q]
    
            # [display] refinement step in STOR simulation
            print("C_OR = ", np.round(100*C_OR_range_STOR[q], decimals = 1), "%")
            
            # [loop] across refinement steps to increase accuracy
            for n_refine_STOR in range(N_refine_STOR):
                
                # [initialize] range for current refinement step; each step increases accuracy by one digit
                if n_refine_STOR == 0:
                    f_demand_STOR_start = f_init_STOR_start
                    f_demand_STOR_step = f_init_STOR_step
                    f_demand_STOR_end = f_init_STOR_end
                elif n_refine_STOR > 0:
                    f_demand_STOR_start = f_demand_opt_STOR - f_demand_STOR_step
                    f_demand_STOR_end = f_demand_opt_STOR + f_demand_STOR_step
                    f_demand_STOR_step = f_demand_STOR_step/10
                
                f_demand_STOR = np.arange(f_demand_STOR_start, f_demand_STOR_end + f_demand_STOR_step, f_demand_STOR_step)
    
                # [preallocate] psi (eq. S21)
                psi_STOR = np.full([len(f_demand_STOR)], np.nan)
                
                
                # [loop] to find optimal values of E_solar and E_wind (eq. S25) by locating minimum in psi (eq. S22)
                for f in range(len(f_demand_STOR)):
                    
                    # [display] progress within each refinement step in STOR simulation
                    print("refinement step", n_refine_STOR + 1, "/", N_refine_STOR, "> scanning:", np.floor(100*(f + 1)/len(f_demand_STOR)), "%")
    
                    # [initialize] realistic value of total SW power (MWh/year) so that we can loop over realistic values of c_solar and c_wind (eq. S25)
                    E_SW_loop_STOR = np.mean(E_hydro_CONV_stable_yearly[:,HPP])*f_demand_STOR[f]*np.ones(shape = (len(E_hydro_CONV_stable_yearly[:,HPP])))
                    
                    # [preallocate] stable hydropower generation P_stable in MW (see explanation below eq. S19)
                    P_STOR_hydro_stable_hourly[:,:,HPP] = np.nan
                    
                    # [loop] across all simulation years to identify realistic c_solar and c_wind values
                    for y in range(len(simulation_years)):
    
                        # [read] vector with hours in each year
                        hrs_year = range(int(hrs_byyear[y]))
                        
                        # [calculate] determine realistic amount of SW capacity in MW (c_solar, c_wind) corresponding to generation equal to E_SW_loop_STOR
                        E_SW_per_MW_STOR_yearly[y,HPP] = np.sum(c_solar_relative[HPP]*CF_solar_hourly[hrs_year,y,HPP] + c_wind_relative[HPP]*CF_wind_hourly[hrs_year,y,HPP])
                        c_multiplier_STOR[y,HPP] = E_SW_loop_STOR[y]/E_SW_per_MW_STOR_yearly[y,HPP]
                        
                    # [loop] perform iterations to get converged estimate of P_stable (see explanation below eq. S19)
                    for x in range(X_max_STOR):
                                        
                        # [calculate] environmentally required outflow (eq. S14)
                        temp_Q_out_STOR = Q_in_nat_av*np.ones(shape = (len(Q_CONV_stable_hourly),len(Q_CONV_stable_hourly[0])))
                        temp_Q_out_STOR[np.isnan(Q_CONV_stable_hourly[:,:,HPP])] = np.nan
                        Q_STOR_stable_hourly[:,:,HPP] = Q_stable_ratio*temp_Q_out_STOR
                        
                        # [initialize] This variable is equal to unity by default, but set to zero in case of extreme droughts forcing a
                        # temporary curtailment on hydropower generation (section S3.1)
                        hydro_STOR_curtailment_factor_hourly[:,:,HPP] = 1
    
                        # [loop] across all simulation years to initialize P_stable (see explanation below eq. S19)
                        for y in range(len(simulation_years)):
                        
                            # [read] vector with hours in each year
                            hrs_year = range(int(hrs_byyear[y]))
    
                            # [initialize] stable hydropower generation P_stable in MW (see explanation below eq. S19)
                            # use estimate P_stable,STOR = (1 - C_OR)*P_stable,CONV as initial guess
                            if x == 0:
                                P_STOR_inflexible_hourly[hrs_year,y,HPP] = (np.mean(c_multiplier_STOR[:,HPP])*c_solar_relative[HPP]*CF_solar_hourly[hrs_year,y,HPP] + np.mean(c_multiplier_STOR[:,HPP])*c_wind_relative[HPP]*CF_wind_hourly[hrs_year,y,HPP] + Q_stable_ratio*np.nanmean(P_CONV_hydro_stable_hourly[:,:,HPP]))
                            elif x > 0:
                                # use estimate P_stable,STOR from previous iteration
                                P_STOR_inflexible_hourly[hrs_year,y,HPP] = (np.mean(c_multiplier_STOR[:,HPP])*c_solar_relative[HPP]*CF_solar_hourly[hrs_year,y,HPP] + np.mean(c_multiplier_STOR[:,HPP])*c_wind_relative[HPP]*CF_wind_hourly[hrs_year,y,HPP] + P_STOR_hydro_stable_hourly[hrs_year,y,HPP])
                            
                        # [calculate] P_load according to constraints on overproduction (eq. S11)
                        P_load_STOR = np.nanpercentile(P_STOR_inflexible_hourly[:,:,HPP],f_size)
                        
                        
                        # [loop] across all simulation years to perform optimization
                        for y in range(len(simulation_years)):
                            
                            # [read] vector with hours in each year
                            hrs_year = range(int(hrs_byyear[y]))
                            
                            # [calculate] hourly load time series in MW (eq. S10)
                            L_STOR_hourly[hrs_year,y,HPP] = P_load_STOR*L_norm[hrs_year,y,HPP]
                            
                            # [calculate] load difference P_d (eq. S9)
                            P_STOR_difference_hourly[hrs_year,y,HPP] = P_STOR_inflexible_hourly[hrs_year,y,HPP] - L_STOR_hourly[hrs_year,y,HPP]
                            
                            # [initialize] initial values of volume (m^3), area (m^2), hydraulic head (m) and ramp restrictions (MW/hr) for each simulation year
                            if y == 0:
                                
                                V_STOR_hourly_upper[0,y,HPP] = V_CONV_hourly[0,y,HPP]
                                V_STOR_hourly_lower[0,y,HPP] = V_lower_max[HPP]*f_opt
                                A_STOR_hourly_upper[0,y,HPP] = A_CONV_hourly[0,y,HPP]
                                h_STOR_hourly[0,y,HPP] = h_CONV_hourly[0,y,HPP]
                                
                                # [calculate] ramping constraint (eq. S16)
                                temp_sgn_turb = 1
                                P_STOR_ramp_restr_hourly[0,y,HPP] = P_r_turb[HPP]*dP_ramp_turb*mins_hr
                                
                                # [calculate] ramping constraint for pump (eq. S37)
                                temp_sgn_pump = 1;
                                P_STOR_ramp_restr_pump_hourly[0,y,HPP] = P_r_pump[HPP]*dP_ramp_pump*mins_hr
                                                            
                            else:
                                
                                temp = V_STOR_hourly_upper[:,y-1,HPP]
                                temp = temp[np.isfinite(temp)]
                                V_STOR_hourly_upper[0,y,HPP] = temp[-1]
                                
                                temp = V_STOR_hourly_lower[:,y-1,HPP]
                                temp = temp[np.isfinite(temp)]
                                V_STOR_hourly_lower[0,y,HPP] = temp[-1]
                
                                temp = A_STOR_hourly_upper[:,y-1,HPP]
                                temp = temp[np.isfinite(temp)]
                                A_STOR_hourly_upper[0,y,HPP] = temp[-1]
                
                                temp = h_STOR_hourly[:,y-1,HPP]
                                temp = temp[np.isfinite(temp)]
                                h_STOR_hourly[0,y,HPP] = temp[-1]
                                
                                
                                # [calculate] ramping constraint (eq. S16)
                                temp = P_STOR_hydro_flexible_hourly[:,y-1,HPP]
                                temp = temp[np.isfinite(temp)]
                                temp_P_difference = P_STOR_difference_hourly[:,y-1,HPP]
                                temp_P_difference = temp_P_difference[np.isfinite(temp_P_difference)]
                                # [calculate] whether ramping up (temp_sgn = 1) or down (temp_sgn = -1)
                                if P_STOR_difference_hourly[0,y,HPP] - temp_P_difference[-1] < 0:
                                    temp_sgn_turb = 1
                                else:
                                    temp_sgn_turb = -1
                                P_STOR_ramp_restr_hourly[0,y,HPP] = temp[-1] + temp_sgn_turb*P_r_turb[HPP]*dP_ramp_turb*mins_hr
                                if P_STOR_ramp_restr_hourly[0,y,HPP] < 0:
                                    P_STOR_ramp_restr_hourly[0,y,HPP] = 0
                                    
                                    
                                # [calculate] ramping constraint for pump (eq. S37)
                                temp = P_STOR_pump_hourly[:,y-1,HPP]
                                temp = temp[np.isfinite(temp)]
                                # [calculate] whether ramping up (temp_sgn = 1) or down (temp_sgn = -1)
                                if P_STOR_difference_hourly[0,y,HPP] - temp_P_difference[-1] < 0:
                                    temp_sgn_pump = -1
                                else:
                                    temp_sgn_pump = 1
                                P_STOR_ramp_restr_pump_hourly[0,y,HPP] = temp[-1] + temp_sgn_pump*P_r_pump[HPP]*dP_ramp_pump*mins_hr
                                if P_STOR_ramp_restr_pump_hourly[0,y,HPP] < 0:
                                    P_STOR_ramp_restr_pump_hourly[0,y,HPP] = 0
                                
                            # [loop] over all time steps in each simulation year to calculate reservoir dynamics and hydropower generation
                            for n in hrs_year:
                                
                                # [check] stable outflow is reduced to zero in case of droughts
                                Q_STOR_stable_hourly[n,y,HPP] = Q_STOR_stable_hourly[n,y,HPP] * hydro_STOR_curtailment_factor_hourly[n,y,HPP]
                                
                                # [calculate] flexible hydropower generation in MW (eq. S16, S17)
                                if P_STOR_difference_hourly[n,y,HPP] < 0:
                                    Q_STOR_pot_turb_flexible[n,y,HPP] = np.max([0, Q_max_turb[HPP] - Q_STOR_stable_hourly[n,y,HPP]]) * hydro_STOR_curtailment_factor_hourly[n,y,HPP]
                                    # [calculate] if ramping up
                                    if temp_sgn_turb == 1:
                                        P_STOR_hydro_flexible_hourly[n,y,HPP] = np.min([Q_STOR_pot_turb_flexible[n,y,HPP]*eta_turb*rho*g*h_STOR_hourly[n,y,HPP]/10**6, np.min([np.abs(P_STOR_difference_hourly[n,y,HPP]), P_STOR_ramp_restr_hourly[n,y,HPP]]) ])
                                    # [calculate] if ramping down
                                    elif temp_sgn_turb == -1:
                                        P_STOR_hydro_flexible_hourly[n,y,HPP] = np.min([Q_STOR_pot_turb_flexible[n,y,HPP]*eta_turb*rho*g*h_STOR_hourly[n,y,HPP]/10**6, np.max([np.abs(P_STOR_difference_hourly[n,y,HPP]), P_STOR_ramp_restr_hourly[n,y,HPP]]) ])
                                    # [calculate] if P_d < 0 pumping is not performed (eq. S37)
                                    P_STOR_pump_hourly[n,y,HPP] = 0
                                
                                # [calculate] pumping power in cases of surpluses (eq. S37, S38)
                                if P_STOR_difference_hourly[n,y,HPP] >= 0:
                                    if V_STOR_hourly_upper[n,y,HPP]/V_max[HPP] < f_spill:
                                        Q_STOR_pot_pump_hourly[n,y,HPP] = np.min([V_STOR_hourly_lower[n,y,HPP]/secs_hr, Q_max_pump[HPP]])
                                        # [calculate] if ramping up
                                        if temp_sgn_pump == 1:
                                            P_STOR_pump_hourly[n,y,HPP] = np.min([Q_STOR_pot_pump_hourly[n,y,HPP]*eta_pump**(-1)*rho*g*h_STOR_hourly[n,y,HPP]/10**6, np.min([np.abs(P_STOR_difference_hourly[n,y,HPP]), P_STOR_ramp_restr_pump_hourly[n,y,HPP]]) ])
                                        # [calculate] if ramping down
                                        elif temp_sgn_pump == -1:
                                            P_STOR_pump_hourly[n,y,HPP] = np.min([Q_STOR_pot_pump_hourly[n,y,HPP]*eta_pump**(-1)*rho*g*h_STOR_hourly[n,y,HPP]/10**6, np.max([np.abs(P_STOR_difference_hourly[n,y,HPP]), P_STOR_ramp_restr_pump_hourly[n,y,HPP]]) ])
                                    else:
                                        P_STOR_pump_hourly[n,y,HPP] = 0
                                    # [check] flexible hydropower generation is zero when P_d >= 0 (eq. S16)
                                    P_STOR_hydro_flexible_hourly[n,y,HPP] = 0
                                    
                                # [calculate] stable hydropower generation in MW (eq. S15)
                                Q_pot_turb_STOR = np.min([Q_STOR_stable_hourly[n,y,HPP], Q_max_turb[HPP]])
                                P_STOR_hydro_stable_hourly[n,y,HPP] = Q_pot_turb_STOR*eta_turb*rho*g*h_STOR_hourly[n,y,HPP]/10**6
    
                                # [calculate] flexible turbined flow (eq. S18) and pumped flow (eq. 39) in m^3/s
                                if h_STOR_hourly[n,y,HPP] > 0:
                                    Q_STOR_flexible_hourly[n,y,HPP] = P_STOR_hydro_flexible_hourly[n,y,HPP]/(eta_turb*rho*g*h_STOR_hourly[n,y,HPP])*10**6
                                    Q_STOR_pump_hourly[n,y,HPP] = P_STOR_pump_hourly[n,y,HPP]/(eta_pump**(-1)*rho*g*h_STOR_hourly[n,y,HPP])*10**6
                                else:
                                    # [check] cannot be negative
                                    h_STOR_hourly[n,y,HPP] = 0
                                    Q_STOR_flexible_hourly[n,y,HPP] = 0
                                    Q_STOR_pump_hourly[n,y,HPP] = 0
                                
                                # [calculate] spilling component of upper reservoir in m^3/s (eq. S19)
                                if V_STOR_hourly_upper[n,y,HPP]/V_max[HPP] < f_spill:
                                    Q_STOR_spill_hourly_upper[n,y,HPP] = 0
                                else:
                                    Q_STOR_spill_hourly_upper[n,y,HPP] = (Q_in_frac_hourly[n,y,HPP] + (precipitation_flux_hourly[n,y,HPP] - evaporation_flux_hourly[n,y,HPP])*A_STOR_hourly_upper[n,y,HPP]/rho)*(1 + mu) - Q_STOR_stable_hourly[n,y,HPP] - Q_STOR_flexible_hourly[n,y,HPP]
                                    # [check] spilling component cannot be negative (eq. S7)
                                    if Q_STOR_spill_hourly_upper[n,y,HPP] < 0:
                                        Q_STOR_spill_hourly_upper[n,y,HPP] = 0
                                
                                # [calculate] spilling component of lower reservoir in m^3/s (eq. S40)
                                if (V_lower_max[HPP] - V_STOR_hourly_lower[n,y,HPP])/secs_hr < Q_STOR_flexible_hourly[n,y,HPP]:
                                    Q_STOR_spill_hourly_lower[n,y,HPP] = Q_STOR_flexible_hourly[n,y,HPP] - (V_lower_max[HPP] - V_STOR_hourly_lower[n,y,HPP])/secs_hr
                                elif (V_lower_max[HPP] - V_STOR_hourly_lower[n,y,HPP])/secs_hr >= Q_STOR_flexible_hourly[n,y,HPP]:
                                    Q_STOR_spill_hourly_lower[n,y,HPP] = 0
                                
                                # [calculate] total net outflow in m^3/s (eq. S36)
                                Q_STOR_out_hourly[n,y,HPP] = Q_STOR_stable_hourly[n,y,HPP] + Q_STOR_spill_hourly_upper[n,y,HPP] + Q_STOR_spill_hourly_lower[n,y,HPP]
                                                            
                                # [calculate] reservoir volume in m^3 at next time step (eq. S34, S35)
                                V_STOR_hourly_upper[n+1,y,HPP] = V_STOR_hourly_upper[n,y,HPP] + (Q_in_frac_hourly[n,y,HPP] - Q_STOR_stable_hourly[n,y,HPP] - Q_STOR_flexible_hourly[n,y,HPP] - Q_STOR_spill_hourly_upper[n,y,HPP] + Q_STOR_pump_hourly[n,y,HPP] + (precipitation_flux_hourly[n,y,HPP] - evaporation_flux_hourly[n,y,HPP])*A_STOR_hourly_upper[n,y,HPP]/rho)*secs_hr
                                V_STOR_hourly_lower[n+1,y,HPP] = V_STOR_hourly_lower[n,y,HPP] + (Q_STOR_flexible_hourly[n,y,HPP] - Q_STOR_pump_hourly[n,y,HPP] - Q_STOR_spill_hourly_lower[n,y,HPP])*secs_hr
                                
                                # [check] prevent unreal values when lake levels drop low
                                if V_STOR_hourly_upper[n+1,y,HPP] < 0:
                                    Q_STOR_stable_hourly[n,y,HPP] = 0
                                    P_STOR_hydro_stable_hourly[n,y,HPP] = 0
                                    Q_STOR_flexible_hourly[n,y,HPP] = 0
                                    P_STOR_hydro_flexible_hourly[n,y,HPP] = 0
                                    Q_STOR_out_hourly[n,y,HPP] = Q_STOR_stable_hourly[n,y,HPP] + Q_STOR_flexible_hourly[n,y,HPP] + Q_STOR_spill_hourly_upper[n,y,HPP] + Q_in_RoR_hourly[n,y,HPP]
                                    A_STOR_hourly_upper[n,y,HPP] = 0
                                    V_STOR_hourly_upper[n+1,y,HPP] = V_STOR_hourly_upper[n,y,HPP] + (Q_in_frac_hourly[n,y,HPP] - Q_STOR_stable_hourly[n,y,HPP] - Q_STOR_flexible_hourly[n,y,HPP] - Q_STOR_spill_hourly_upper[n,y,HPP] + Q_STOR_pump_hourly[n,y,HPP] + (precipitation_flux_hourly[n,y,HPP] - evaporation_flux_hourly[n,y,HPP])*A_STOR_hourly_upper[n,y,HPP]/rho)*secs_hr
                                
                                # [calculate] reservoir lake area in m^2 and hydraulic head in m from bathymetric relationship
                                h_temp = np.where(abs(calibrate_volume[:,HPP] - V_STOR_hourly_upper[n+1,y,HPP]) == min(abs(calibrate_volume[:,HPP] - V_STOR_hourly_upper[n+1,y,HPP])))[0][0]
                                A_STOR_hourly_upper[n+1,y,HPP] = calibrate_area[h_temp,HPP]
                                h_STOR_hourly[n+1,y,HPP] = calibrate_head[h_temp,HPP]
                                
                                # [calculate] ramp rate restrictions (MW attainable) at next time step (for turbine) (eq. S16)
                                if n < len(hrs_year) - 1:
                                    if (P_STOR_difference_hourly[n+1,y,HPP] - P_STOR_difference_hourly[n,y,HPP]) < 0:
                                        temp_sgn_turb = 1
                                    else:
                                        temp_sgn_turb = -1
                                    P_STOR_ramp_restr_hourly[n+1,y,HPP] = P_STOR_hydro_flexible_hourly[n,y,HPP] + temp_sgn_turb*P_r_turb[HPP]*dP_ramp_turb*mins_hr
                                    if P_STOR_ramp_restr_hourly[n+1,y,HPP] < 0:
                                        P_STOR_ramp_restr_hourly[n+1,y,HPP] = 0
                                
                                # [calculate] ramp rate restrictions (MW attainable) at next time step (for pump) (eq. S37)
                                if n < len(hrs_year) - 1:
                                    if (P_STOR_difference_hourly[n+1,y,HPP] - P_STOR_difference_hourly[n,y,HPP]) < 0:
                                        temp_sgn_pump = -1
                                    else:
                                        temp_sgn_pump = 1
                                    P_STOR_ramp_restr_pump_hourly[n+1,y,HPP] = P_STOR_pump_hourly[n,y,HPP] + temp_sgn_pump*P_r_pump[HPP]*dP_ramp_pump*mins_hr
                                    if P_STOR_ramp_restr_pump_hourly[n+1,y,HPP] < 0:
                                        P_STOR_ramp_restr_pump_hourly[n+1,y,HPP] = 0
                                
                                
                                # [calculate] whether lake levels have dropped so low as to require hydropower curtailment
                                # curtail hydropower generation in case water levels have dropped below f_stop*V_max
                                # (see section S3.1)
                                if V_STOR_hourly_upper[n+1,y,HPP] < f_stop*V_max[HPP]:
                                    if n < len(hrs_year) - 1:
                                        hydro_STOR_curtailment_factor_hourly[n+1,y,HPP] = 0
                                    elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                                        hydro_STOR_curtailment_factor_hourly[0,y+1,HPP] = 0
                                        
                                # [calculate] restart hydropower generation if reservoir levels have recovered
                                # (see section S3.1)
                                if hydro_STOR_curtailment_factor_hourly[n,y,HPP] == 0 and V_STOR_hourly_upper[n+1,y,HPP] > f_restart*V_max[HPP]:
                                    if n < len(hrs_year) - 1:
                                        hydro_STOR_curtailment_factor_hourly[n+1,y,HPP] = 1
                                    elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                                        hydro_STOR_curtailment_factor_hourly[0,y+1,HPP] = 1
                                elif hydro_STOR_curtailment_factor_hourly[n,y,HPP] == 0 and V_STOR_hourly_upper[n+1,y,HPP] <= f_restart*V_max[HPP]:
                                    if n < len(hrs_year) - 1:
                                        hydro_STOR_curtailment_factor_hourly[n+1,y,HPP] = 0
                                    elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                                        hydro_STOR_curtailment_factor_hourly[0,y+1,HPP] = 0
                                
                            
                        
                    # [arrange] complete time series of water volume for eq. S20
                    for y in range(len(simulation_years)):
                        V_STOR_hourly_upper[int(hrs_byyear[y]),y,HPP] = np.nan
                    
                    temp_volume_upper_STOR_series = V_STOR_hourly_upper[:,:,HPP]
                    temp_volume_upper_STOR_series = (np.transpose(temp_volume_upper_STOR_series)).ravel()
                    temp_volume_upper_STOR_series = temp_volume_upper_STOR_series[np.isfinite(temp_volume_upper_STOR_series)]
                    
                    # [calculate] deviation between CONV and STOR reservoir dynamics (eq. S21)
                    psi_STOR[f] = np.mean(np.abs(temp_volume_upper_STOR_series - V_CONV_series_hourly[:,HPP]))/np.mean(V_CONV_series_hourly[:,HPP])
                    
                    # [check] see explanation below eq. S21: if droughts occur in CONV, STOR should have no MORE days of curtailed flow than CONV ...
                    # and curtailed flow should occur in less than 50% of the years in the simulation, so median yearly statistics represent normal operation
                    if np.nanmin(V_CONV_hourly[:,:,HPP]) < f_stop*V_max[HPP] and (np.nansum(Q_STOR_out_hourly[:,:,HPP] == 0) > np.nansum(Q_CONV_out_hourly[:,:,HPP] == 0) or np.sum(np.sum(Q_STOR_out_hourly[:,:,HPP] - Q_in_RoR_hourly[:,:,HPP] == 0, axis = 0) > 0) > np.floor(len(simulation_years)/2)):
                        psi_STOR[f] = np.nan
                        # [check] if droughts do not occur in CONV, then neither should they in STOR
                    elif np.nanmin(V_CONV_hourly[:,:,HPP]) >= f_stop*V_max[HPP] and np.nanmin(V_STOR_hourly_upper[:,:,HPP]) < f_stop*V_max[HPP]:
                        psi_STOR[f] = np.nan
                    
                # psi_STOR
                
                # [identify] minimum in psi (eq. S21)
                if np.sum(np.isnan(psi_STOR)) == len(psi_STOR) and f_demand_STOR[0] == 0:
                    f_demand_opt_STOR = 0
                    psi_STOR_opt = 0
                    break
                else:
                    crossing_STOR = np.where(psi_STOR == min(psi_STOR))[0][0]
                    f_demand_opt_STOR = f_demand_STOR[crossing_STOR]
                    psi_STOR_opt = np.abs(psi_STOR[crossing_STOR])
                
                # [check] prevent negative results
                if f_demand_opt_STOR == 0:
                    f_demand_opt_STOR = f_demand_STOR[crossing_STOR + 1]
                    psi_STOR_opt = np.abs(psi_STOR[crossing_STOR + 1])
                    
                # [check] determine if psi is low enough for this to qualify as optimum solution
                if psi_STOR_opt < psi_min_threshold:
                    break
                
                # [check] if range in which to identify ELCC is adequate
                if f_demand_opt_STOR == f_demand_STOR[-1]:
                    print("Warning: parameter f_init_STOR_end likely set too low")
                
            # [initialize] optimal value of total SW power (MWh/year) so that we can calculate optimal c_solar and c_wind (eq. S25)
            E_SW_loop_STOR_opt[HPP] = np.mean(E_hydro_CONV_stable_yearly[:,HPP])*f_demand_opt_STOR
            
            
            ###############################################################
            ############----------- STOR optimized ------------############
            ###############################################################
            
            # [display]
            print("(v) found optimum STOR solution - saving all variables")
            
            # [preallocate] to test convergence towards P_stable (see explanation below eq. S19)
            convergence_test_STOR = np.zeros(shape = (X_max_STOR))
            
            # [loop] across all simulation years to identify realistic c_solar and c_wind values
            for y in range(len(simulation_years)):

                # [read] vector with hours in each year
                hrs_year = range(int(hrs_byyear[y]))
                
                # [calculate] determine realistic amount of SW capacity in MW (c_solar, c_wind) corresponding to generation equal to E_SW_loop_STOR
                E_SW_per_MW_STOR_yearly[y,HPP] = np.sum(c_solar_relative[HPP]*CF_solar_hourly[hrs_year,y,HPP] + c_wind_relative[HPP]*CF_wind_hourly[hrs_year,y,HPP])
                c_multiplier_STOR[y,HPP] = E_SW_loop_STOR_opt[HPP]/E_SW_per_MW_STOR_yearly[y,HPP]
                
                
            # [loop] perform iterations to get converged estimate of P_stable (see explanation below eq. S19)
            for x in range(X_max_STOR):
                                
                # [calculate] environmentally required outflow (eq. S14)
                temp_Q_out_STOR = Q_in_nat_av*np.ones(shape = (len(Q_CONV_stable_hourly),len(Q_CONV_stable_hourly[0])))
                temp_Q_out_STOR[np.isnan(Q_CONV_stable_hourly[:,:,HPP])] = np.nan
                Q_STOR_stable_hourly[:,:,HPP] = Q_stable_ratio*temp_Q_out_STOR
                
                # [initialize] This variable is equal to unity by default, but set to zero in case of extreme droughts forcing a
                # temporary curtailment on hydropower generation (section S3.1)
                hydro_STOR_curtailment_factor_hourly[:,:,HPP] = 1

                # [loop] across all simulation years to initialize P_stable (see explanation below eq. S19)
                for y in range(len(simulation_years)):
                
                    # [read] vector with hours in each year
                    hrs_year = range(int(hrs_byyear[y]))

                    # [initialize] stable hydropower generation P_stable in MW (see explanation below eq. S19)
                    # use estimate P_stable,STOR = (1 - C_OR)*P_stable,CONV as initial guess
                    if x == 0:
                        P_STOR_inflexible_hourly[hrs_year,y,HPP] = (np.mean(c_multiplier_STOR[:,HPP])*c_solar_relative[HPP]*CF_solar_hourly[hrs_year,y,HPP] + np.mean(c_multiplier_STOR[:,HPP])*c_wind_relative[HPP]*CF_wind_hourly[hrs_year,y,HPP] + Q_stable_ratio*np.nanmean(P_CONV_hydro_stable_hourly[:,:,HPP]))
                    elif x > 0:
                        # use estimate P_stable,STOR from previous iteration
                        P_STOR_inflexible_hourly[hrs_year,y,HPP] = (np.mean(c_multiplier_STOR[:,HPP])*c_solar_relative[HPP]*CF_solar_hourly[hrs_year,y,HPP] + np.mean(c_multiplier_STOR[:,HPP])*c_wind_relative[HPP]*CF_wind_hourly[hrs_year,y,HPP] + P_STOR_hydro_stable_hourly[hrs_year,y,HPP])
                    
                    # [calculate] total solar and wind power generation by hour (eq. S12)
                    P_STOR_solar_hourly[hrs_year,y,HPP] = np.mean(c_multiplier_STOR[:,HPP])*c_solar_relative[HPP]*CF_solar_hourly[hrs_year,y,HPP]
                    P_STOR_wind_hourly[hrs_year,y,HPP] = np.mean(c_multiplier_STOR[:,HPP])*c_wind_relative[HPP]*CF_wind_hourly[hrs_year,y,HPP]
                
                
                # [calculate] P_load according to constraints on overproduction (eq. S11)
                P_load_STOR = np.nanpercentile(P_STOR_inflexible_hourly[:,:,HPP],f_size)
                
                
                # [loop] across all simulation years to perform optimization
                for y in range(len(simulation_years)):
                    
                    # [read] vector with hours in each year
                    hrs_year = range(int(hrs_byyear[y]))
                    
                    # [calculate] hourly load time series in MW (eq. S10)
                    L_STOR_hourly[hrs_year,y,HPP] = P_load_STOR*L_norm[hrs_year,y,HPP]
                    
                    # [calculate] load difference P_d (eq. S9)
                    P_STOR_difference_hourly[hrs_year,y,HPP] = P_STOR_inflexible_hourly[hrs_year,y,HPP] - L_STOR_hourly[hrs_year,y,HPP]
                    
                    # [initialize] initial values of volume (m^3), area (m^2), hydraulic head (m) and ramp restrictions (MW/hr) for each simulation year
                    if y == 0:
                        
                        V_STOR_hourly_upper[0,y,HPP] = V_CONV_hourly[0,y,HPP]
                        V_STOR_hourly_lower[0,y,HPP] = V_lower_max[HPP]*f_opt
                        A_STOR_hourly_upper[0,y,HPP] = A_CONV_hourly[0,y,HPP]
                        h_STOR_hourly[0,y,HPP] = h_CONV_hourly[0,y,HPP]
                        
                        # [calculate] ramping constraint (eq. S16)
                        temp_sgn_turb = 1
                        P_STOR_ramp_restr_hourly[0,y,HPP] = P_r_turb[HPP]*dP_ramp_turb*mins_hr
                        
                        # [calculate] ramping constraint for pump (eq. S37)
                        temp_sgn_pump = 1;
                        P_STOR_ramp_restr_pump_hourly[0,y,HPP] = P_r_pump[HPP]*dP_ramp_pump*mins_hr
                                                    
                    else:
                        
                        temp = V_STOR_hourly_upper[:,y-1,HPP]
                        temp = temp[np.isfinite(temp)]
                        V_STOR_hourly_upper[0,y,HPP] = temp[-1]
                        
                        temp = V_STOR_hourly_lower[:,y-1,HPP]
                        temp = temp[np.isfinite(temp)]
                        V_STOR_hourly_lower[0,y,HPP] = temp[-1]
        
                        temp = A_STOR_hourly_upper[:,y-1,HPP]
                        temp = temp[np.isfinite(temp)]
                        A_STOR_hourly_upper[0,y,HPP] = temp[-1]
        
                        temp = h_STOR_hourly[:,y-1,HPP]
                        temp = temp[np.isfinite(temp)]
                        h_STOR_hourly[0,y,HPP] = temp[-1]
                        
                        
                        # [calculate] ramping constraint (eq. S16)
                        temp = P_STOR_hydro_flexible_hourly[:,y-1,HPP]
                        temp = temp[np.isfinite(temp)]
                        temp_P_difference = P_STOR_difference_hourly[:,y-1,HPP]
                        temp_P_difference = temp_P_difference[np.isfinite(temp_P_difference)]
                        # [calculate] whether ramping up (temp_sgn = 1) or down (temp_sgn = -1)
                        if P_STOR_difference_hourly[0,y,HPP] - temp_P_difference[-1] < 0:
                            temp_sgn_turb = 1
                        else:
                            temp_sgn_turb = -1
                        P_STOR_ramp_restr_hourly[0,y,HPP] = temp[-1] + temp_sgn_turb*P_r_turb[HPP]*dP_ramp_turb*mins_hr
                        if P_STOR_ramp_restr_hourly[0,y,HPP] < 0:
                            P_STOR_ramp_restr_hourly[0,y,HPP] = 0
                            
                            
                        # [calculate] ramping constraint for pump (eq. S37)
                        temp = P_STOR_pump_hourly[:,y-1,HPP]
                        temp = temp[np.isfinite(temp)]
                        # [calculate] whether ramping up (temp_sgn = 1) or down (temp_sgn = -1)
                        if P_STOR_difference_hourly[0,y,HPP] - temp_P_difference[-1] < 0:
                            temp_sgn_pump = -1
                        else:
                            temp_sgn_pump = 1
                        P_STOR_ramp_restr_pump_hourly[0,y,HPP] = temp[-1] + temp_sgn_pump*P_r_pump[HPP]*dP_ramp_pump*mins_hr
                        if P_STOR_ramp_restr_pump_hourly[0,y,HPP] < 0:
                            P_STOR_ramp_restr_pump_hourly[0,y,HPP] = 0
                        
                    # [loop] over all time steps in each simulation year to calculate reservoir dynamics and hydropower generation
                    for n in hrs_year:
                        
                        # [check] stable outflow is reduced to zero in case of droughts
                        Q_STOR_stable_hourly[n,y,HPP] = Q_STOR_stable_hourly[n,y,HPP] * hydro_STOR_curtailment_factor_hourly[n,y,HPP]
                        
                        # [calculate] flexible hydropower generation in MW (eq. S16, S17)
                        if P_STOR_difference_hourly[n,y,HPP] < 0:
                            Q_STOR_pot_turb_flexible[n,y,HPP] = np.max([0, Q_max_turb[HPP] - Q_STOR_stable_hourly[n,y,HPP]]) * hydro_STOR_curtailment_factor_hourly[n,y,HPP]
                            # [calculate] if ramping up
                            if temp_sgn_turb == 1:
                                P_STOR_hydro_flexible_hourly[n,y,HPP] = np.min([Q_STOR_pot_turb_flexible[n,y,HPP]*eta_turb*rho*g*h_STOR_hourly[n,y,HPP]/10**6, np.min([np.abs(P_STOR_difference_hourly[n,y,HPP]), P_STOR_ramp_restr_hourly[n,y,HPP]]) ])
                            # [calculate] if ramping down
                            elif temp_sgn_turb == -1:
                                P_STOR_hydro_flexible_hourly[n,y,HPP] = np.min([Q_STOR_pot_turb_flexible[n,y,HPP]*eta_turb*rho*g*h_STOR_hourly[n,y,HPP]/10**6, np.max([np.abs(P_STOR_difference_hourly[n,y,HPP]), P_STOR_ramp_restr_hourly[n,y,HPP]]) ])
                            # [calculate] if P_d < 0 pumping is not performed (eq. S37)
                            P_STOR_pump_hourly[n,y,HPP] = 0
                        
                        # [calculate] pumping power in cases of surpluses (eq. S37, S38)
                        if P_STOR_difference_hourly[n,y,HPP] >= 0:
                            if V_STOR_hourly_upper[n,y,HPP]/V_max[HPP] < f_spill:
                                Q_STOR_pot_pump_hourly[n,y,HPP] = np.min([V_STOR_hourly_lower[n,y,HPP]/secs_hr, Q_max_pump[HPP]])
                                # [calculate] if ramping up
                                if temp_sgn_pump == 1:
                                    P_STOR_pump_hourly[n,y,HPP] = np.min([Q_STOR_pot_pump_hourly[n,y,HPP]*eta_pump**(-1)*rho*g*h_STOR_hourly[n,y,HPP]/10**6, np.min([np.abs(P_STOR_difference_hourly[n,y,HPP]), P_STOR_ramp_restr_pump_hourly[n,y,HPP]]) ])
                                # [calculate] if ramping down
                                elif temp_sgn_pump == -1:
                                    P_STOR_pump_hourly[n,y,HPP] = np.min([Q_STOR_pot_pump_hourly[n,y,HPP]*eta_pump**(-1)*rho*g*h_STOR_hourly[n,y,HPP]/10**6, np.max([np.abs(P_STOR_difference_hourly[n,y,HPP]), P_STOR_ramp_restr_pump_hourly[n,y,HPP]]) ])
                            else:
                                P_STOR_pump_hourly[n,y,HPP] = 0
                            # [check] flexible hydropower generation is zero when P_d >= 0 (eq. S16)
                            P_STOR_hydro_flexible_hourly[n,y,HPP] = 0
                            
                        # [calculate] stable hydropower generation in MW (eq. S15)
                        Q_pot_turb_STOR = np.min([Q_STOR_stable_hourly[n,y,HPP], Q_max_turb[HPP]])
                        P_STOR_hydro_stable_hourly[n,y,HPP] = Q_pot_turb_STOR*eta_turb*rho*g*h_STOR_hourly[n,y,HPP]/10**6

                        # [calculate] flexible turbined flow (eq. S18) and pumped flow (eq. 39) in m^3/s
                        if h_STOR_hourly[n,y,HPP] > 0:
                            Q_STOR_flexible_hourly[n,y,HPP] = P_STOR_hydro_flexible_hourly[n,y,HPP]/(eta_turb*rho*g*h_STOR_hourly[n,y,HPP])*10**6
                            Q_STOR_pump_hourly[n,y,HPP] = P_STOR_pump_hourly[n,y,HPP]/(eta_pump**(-1)*rho*g*h_STOR_hourly[n,y,HPP])*10**6
                        else:
                            # [check] cannot be negative
                            h_STOR_hourly[n,y,HPP] = 0
                            Q_STOR_flexible_hourly[n,y,HPP] = 0
                            Q_STOR_pump_hourly[n,y,HPP] = 0
                        
                        # [calculate] spilling component of upper reservoir in m^3/s (eq. S19)
                        if V_STOR_hourly_upper[n,y,HPP]/V_max[HPP] < f_spill:
                            Q_STOR_spill_hourly_upper[n,y,HPP] = 0
                        else:
                            Q_STOR_spill_hourly_upper[n,y,HPP] = (Q_in_frac_hourly[n,y,HPP] + (precipitation_flux_hourly[n,y,HPP] - evaporation_flux_hourly[n,y,HPP])*A_STOR_hourly_upper[n,y,HPP]/rho)*(1 + mu) - Q_STOR_stable_hourly[n,y,HPP] - Q_STOR_flexible_hourly[n,y,HPP]
                            # [check] spilling component cannot be negative (eq. S7)
                            if Q_STOR_spill_hourly_upper[n,y,HPP] < 0:
                                Q_STOR_spill_hourly_upper[n,y,HPP] = 0
                        
                        # [calculate] spilling component of lower reservoir in m^3/s (eq. S40)
                        if (V_lower_max[HPP] - V_STOR_hourly_lower[n,y,HPP])/secs_hr < Q_STOR_flexible_hourly[n,y,HPP]:
                            Q_STOR_spill_hourly_lower[n,y,HPP] = Q_STOR_flexible_hourly[n,y,HPP] - (V_lower_max[HPP] - V_STOR_hourly_lower[n,y,HPP])/secs_hr
                        elif (V_lower_max[HPP] - V_STOR_hourly_lower[n,y,HPP])/secs_hr >= Q_STOR_flexible_hourly[n,y,HPP]:
                            Q_STOR_spill_hourly_lower[n,y,HPP] = 0
                        
                        # [calculate] total net outflow in m^3/s (eq. S36)
                        Q_STOR_out_hourly[n,y,HPP] = Q_STOR_stable_hourly[n,y,HPP] + Q_STOR_spill_hourly_upper[n,y,HPP] + Q_STOR_spill_hourly_lower[n,y,HPP]
                                                    
                        # [calculate] reservoir volume in m^3 at next time step (eq. S34, S35)
                        V_STOR_hourly_upper[n+1,y,HPP] = V_STOR_hourly_upper[n,y,HPP] + (Q_in_frac_hourly[n,y,HPP] - Q_STOR_stable_hourly[n,y,HPP] - Q_STOR_flexible_hourly[n,y,HPP] - Q_STOR_spill_hourly_upper[n,y,HPP] + Q_STOR_pump_hourly[n,y,HPP] + (precipitation_flux_hourly[n,y,HPP] - evaporation_flux_hourly[n,y,HPP])*A_STOR_hourly_upper[n,y,HPP]/rho)*secs_hr
                        V_STOR_hourly_lower[n+1,y,HPP] = V_STOR_hourly_lower[n,y,HPP] + (Q_STOR_flexible_hourly[n,y,HPP] - Q_STOR_pump_hourly[n,y,HPP] - Q_STOR_spill_hourly_lower[n,y,HPP])*secs_hr
                        
                        # [check] prevent unreal values when lake levels drop low
                        if V_STOR_hourly_upper[n+1,y,HPP] < 0:
                            Q_STOR_stable_hourly[n,y,HPP] = 0
                            P_STOR_hydro_stable_hourly[n,y,HPP] = 0
                            Q_STOR_flexible_hourly[n,y,HPP] = 0
                            P_STOR_hydro_flexible_hourly[n,y,HPP] = 0
                            Q_STOR_out_hourly[n,y,HPP] = Q_STOR_stable_hourly[n,y,HPP] + Q_STOR_flexible_hourly[n,y,HPP] + Q_STOR_spill_hourly_upper[n,y,HPP] + Q_in_RoR_hourly[n,y,HPP]
                            A_STOR_hourly_upper[n,y,HPP] = 0
                            V_STOR_hourly_upper[n+1,y,HPP] = V_STOR_hourly_upper[n,y,HPP] + (Q_in_frac_hourly[n,y,HPP] - Q_STOR_stable_hourly[n,y,HPP] - Q_STOR_flexible_hourly[n,y,HPP] - Q_STOR_spill_hourly_upper[n,y,HPP] + Q_STOR_pump_hourly[n,y,HPP] + (precipitation_flux_hourly[n,y,HPP] - evaporation_flux_hourly[n,y,HPP])*A_STOR_hourly_upper[n,y,HPP]/rho)*secs_hr
                        
                        # [calculate] reservoir lake area in m^2 and hydraulic head in m from bathymetric relationship
                        h_temp = np.where(abs(calibrate_volume[:,HPP] - V_STOR_hourly_upper[n+1,y,HPP]) == min(abs(calibrate_volume[:,HPP] - V_STOR_hourly_upper[n+1,y,HPP])))[0][0]
                        A_STOR_hourly_upper[n+1,y,HPP] = calibrate_area[h_temp,HPP]
                        h_STOR_hourly[n+1,y,HPP] = calibrate_head[h_temp,HPP]
                        
                        # [calculate] ramp rate restrictions (MW attainable) at next time step (for turbine) (eq. S16)
                        if n < len(hrs_year) - 1:
                            if (P_STOR_difference_hourly[n+1,y,HPP] - P_STOR_difference_hourly[n,y,HPP]) < 0:
                                temp_sgn_turb = 1
                            else:
                                temp_sgn_turb = -1
                            P_STOR_ramp_restr_hourly[n+1,y,HPP] = P_STOR_hydro_flexible_hourly[n,y,HPP] + temp_sgn_turb*P_r_turb[HPP]*dP_ramp_turb*mins_hr
                            if P_STOR_ramp_restr_hourly[n+1,y,HPP] < 0:
                                P_STOR_ramp_restr_hourly[n+1,y,HPP] = 0
                        
                        # [calculate] ramp rate restrictions (MW attainable) at next time step (for pump) (eq. S37)
                        if n < len(hrs_year) - 1:
                            if (P_STOR_difference_hourly[n+1,y,HPP] - P_STOR_difference_hourly[n,y,HPP]) < 0:
                                temp_sgn_pump = -1
                            else:
                                temp_sgn_pump = 1
                            P_STOR_ramp_restr_pump_hourly[n+1,y,HPP] = P_STOR_pump_hourly[n,y,HPP] + temp_sgn_pump*P_r_pump[HPP]*dP_ramp_pump*mins_hr
                            if P_STOR_ramp_restr_pump_hourly[n+1,y,HPP] < 0:
                                P_STOR_ramp_restr_pump_hourly[n+1,y,HPP] = 0
                        
                        
                        # [calculate] whether lake levels have dropped so low as to require hydropower curtailment
                        # curtail hydropower generation in case water levels have dropped below f_stop*V_max
                        # (see section S3.1)
                        if V_STOR_hourly_upper[n+1,y,HPP] < f_stop*V_max[HPP]:
                            if n < len(hrs_year) - 1:
                                hydro_STOR_curtailment_factor_hourly[n+1,y,HPP] = 0
                            elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                                hydro_STOR_curtailment_factor_hourly[0,y+1,HPP] = 0
                                
                        # [calculate] restart hydropower generation if reservoir levels have recovered
                        # (see section S3.1)
                        if hydro_STOR_curtailment_factor_hourly[n,y,HPP] == 0 and V_STOR_hourly_upper[n+1,y,HPP] > f_restart*V_max[HPP]:
                            if n < len(hrs_year) - 1:
                                hydro_STOR_curtailment_factor_hourly[n+1,y,HPP] = 1
                            elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                                hydro_STOR_curtailment_factor_hourly[0,y+1,HPP] = 1
                        elif hydro_STOR_curtailment_factor_hourly[n,y,HPP] == 0 and V_STOR_hourly_upper[n+1,y,HPP] <= f_restart*V_max[HPP]:
                            if n < len(hrs_year) - 1:
                                hydro_STOR_curtailment_factor_hourly[n+1,y,HPP] = 0
                            elif n == len(hrs_year) - 1 and y < len(simulation_years) - 1:
                                hydro_STOR_curtailment_factor_hourly[0,y+1,HPP] = 0
                        
                        
                    # [calculate] total solar and wind power generation under optimal STOR solution in MWh/year (eq. S24)
                    E_solar_STOR_yearly[y,HPP] = np.sum(P_STOR_solar_hourly[hrs_year,y,HPP])
                    E_wind_STOR_yearly[y,HPP] = np.sum(P_STOR_wind_hourly[hrs_year,y,HPP])
                    E_SW_STOR_yearly[y,HPP] = E_solar_STOR_yearly[y,HPP] + E_wind_STOR_yearly[y,HPP]
                    
                    # [calculate] share of solar (= 1 - share of wind) in the resulting SW mix
                    share_SW_STOR[y,HPP] = E_solar_STOR_yearly[y,HPP]/(E_solar_STOR_yearly[y,HPP] + E_wind_STOR_yearly[y,HPP])
                    
                    # [calculate] total flexible hydropower generation under optimal STOR solution in MWh/year (eq. S24)
                    E_hydro_STOR_flexible_yearly[y,HPP] = np.sum(P_STOR_hydro_flexible_hourly[hrs_year,y,HPP])
                                            
                    # [calculate] total stable hydropower generation under optimal STOR solution in MWh/year (eq. S24)
                    E_hydro_STOR_stable_yearly[y,HPP] = np.sum(P_STOR_hydro_stable_hourly[hrs_year,y,HPP])
                    
                    # [calculate] total stable + flexible hydropower generation under optimal STOR solution in MWh/year (eq. S24)
                    E_hydro_STOR_yearly[y,HPP] = E_hydro_STOR_flexible_yearly[y,HPP] + E_hydro_STOR_stable_yearly[y,HPP]
                    
                    # [calculate] total energy pumped up into reservoir in MWh/year
                    E_hydro_STOR_pump_yearly[y,HPP] = np.sum(P_STOR_pump_hourly[hrs_year,y,HPP])*eta_pump
                    
                    # [calculate] total HSW generation under optimal STOR solution in MWh/year
                    temp_power_delivered_STOR = P_STOR_solar_hourly[hrs_year,y,HPP] + P_STOR_wind_hourly[hrs_year,y,HPP] + P_STOR_hydro_stable_hourly[hrs_year,y,HPP] + P_STOR_hydro_flexible_hourly[hrs_year,y,HPP] -1*P_STOR_pump_hourly[hrs_year,y,HPP]
                    
                    # [calculate] net difference between generation and load under optimal STOR solution in MWh/year
                    temp_power_diff_STOR = temp_power_delivered_STOR - L_STOR_hourly[hrs_year,y,HPP]
                    
                    # [calculate] yearly overproduction compared to load under optimal STOR solution in MWh/year
                    E_overproduced_STOR_yearly[y,HPP] = np.abs(np.sum(temp_power_diff_STOR[temp_power_diff_STOR > 0]))
                    
                    # [calculate] yearly SW generation minus overproduction under optimal STOR solution in MWh/year
                    E_SW_STOR_without_overproduction[y,HPP] = E_solar_STOR_yearly[y,HPP] + E_wind_STOR_yearly[y,HPP] - E_hydro_STOR_pump_yearly[y,HPP]/eta_pump - E_overproduced_STOR_yearly[y,HPP]
                    
                    
                    ##### IDENTIFY YEARLY ELCC #####
                    # [calculate] total supplied HSW generation under optimal STOR solution
                    total_power_supply_STOR = P_STOR_hydro_stable_hourly[hrs_year,y,HPP] + P_STOR_hydro_flexible_hourly[hrs_year,y,HPP] - P_STOR_pump_hourly[hrs_year,y,HPP] + np.mean(c_multiplier_STOR[:,HPP])*c_solar_relative[HPP]*CF_solar_hourly[hrs_year,y,HPP] + np.mean(c_multiplier_STOR[:,HPP])*c_wind_relative[HPP]*CF_wind_hourly[hrs_year,y,HPP]
                    N_power_supply_STOR = int(np.ceil(np.max(total_power_supply_STOR)))
                                                              
                    # [preallocate] range in which to identify ELCC
                    P_followed_STOR_range[y,:,HPP] = np.linspace(0,N_power_supply_STOR,N_ELCC)
                    power_unmet_STOR = np.zeros(shape = N_ELCC)
                    
                    # [loop] to identify ELCC under optimal STOR solution
                    for n in range(N_ELCC):
                        temp = total_power_supply_STOR - P_followed_STOR_range[y,n,HPP]*L_norm[hrs_year,y,HPP]
                        if np.abs(np.mean(temp[temp<=0])) > 0:
                            power_unmet_STOR[n] = np.abs(np.sum(temp[temp<=0]))/np.sum(P_followed_STOR_range[y,n,HPP]*L_norm[hrs_year,y,HPP])
                    
                    # [identify] total P_followed given the constraint LOEE_allowed (default zero)
                    N_demand_covered_STOR_temp = np.where(power_unmet_STOR[power_unmet_STOR != np.Inf] > LOEE_allowed)[0][0] - 1
                    if N_demand_covered_STOR_temp.size == 0 or N_demand_covered_STOR_temp == 0:
                        P_followed_STOR_index[y,HPP] = 1
                    else:
                        P_followed_STOR_index[y,HPP] = N_demand_covered_STOR_temp
                    
                    # [identify] hourly time series of L_followed (MW) (eq. S23)
                    L_followed_STOR_hourly[hrs_year,y,HPP] = P_followed_STOR_range[y,int(P_followed_STOR_index[y,HPP]),HPP]*L_norm[hrs_year,y,HPP]
                    
                    # [calculate] ELCC by year (MWh/year) (eq. S23)
                    ELCC_STOR_yearly[y,HPP] = np.sum(L_followed_STOR_hourly[hrs_year,y,HPP])
    
                    # [calculate] difference between ELCC and total HSW generated (excl. RoR component) to obtain Residual Load Duration Curve (RLDC) (eq. S22)
                    L_res_STOR_hourly[hrs_year,y,HPP] = L_followed_STOR_hourly[hrs_year,y,HPP] - total_power_supply_STOR
                    
                    # [arrange] mean fraction of unmet load by month
                    for m in range(months_yr):
                        temp1 = L_res_STOR_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP]
                        temp2 = L_followed_STOR_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP]
                        L_unmet_STOR_frac_bymonth[m,y,HPP] = np.sum(temp1[temp1>0])/np.sum(temp2)
                        
                    # [arrange] yearly average outflow under optimal BAL solution
                    Q_STOR_out_yearly[y,HPP] = np.mean(Q_STOR_out_hourly[hrs_year,y,HPP])
                    
                    
                # [check] to check convergence of solution towards P_stable
                convergence_test_STOR[x] = np.nanmean(P_STOR_hydro_stable_hourly[:,:,HPP])
                
            # [arrange] outflow data by month
            for y in range(len(simulation_years)):
                for m in range(months_yr):
                    Q_STOR_out_monthly[m,y,HPP] = np.mean(Q_STOR_out_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP])
                
            # [arrange] complete time series of water volume, area and levels
            for y in range(len(simulation_years)): 
                V_STOR_hourly_upper[int(hrs_byyear[y]),y,HPP] = np.nan
                V_STOR_hourly_lower[int(hrs_byyear[y]),y,HPP] = np.nan
                A_STOR_hourly_upper[int(hrs_byyear[y]),y,HPP] = np.nan
                h_STOR_hourly[int(hrs_byyear[y]),y,HPP] = np.nan
                
            temp_volume_upper_STOR_series = V_STOR_hourly_upper[:,:,HPP]
            temp_volume_upper_STOR_series = (np.transpose(temp_volume_upper_STOR_series)).ravel()
            temp_volume_upper_STOR_series = temp_volume_upper_STOR_series[np.isfinite(temp_volume_upper_STOR_series)]
            V_STOR_series_hourly_upper[:,HPP] = temp_volume_upper_STOR_series
            
            temp_volume_lower_STOR_series = V_STOR_hourly_lower[:,:,HPP]
            temp_volume_upper_STOR_series = (np.transpose(temp_volume_upper_STOR_series)).ravel()
            temp_volume_upper_STOR_series = temp_volume_upper_STOR_series[np.isfinite(temp_volume_upper_STOR_series)]
            V_STOR_series_hourly_upper[:,HPP] = temp_volume_upper_STOR_series
            
            temp_area_STOR_series = A_STOR_hourly_upper[:,:,HPP]
            temp_area_STOR_series = (np.transpose(temp_area_STOR_series)).ravel()
            temp_area_STOR_series = temp_area_STOR_series[np.isfinite(temp_area_STOR_series)]
            A_STOR_series_hourly_upper[:,HPP] = temp_area_STOR_series
            
            temp_head_STOR_series = h_STOR_hourly[:,:,HPP]
            temp_head_STOR_series = (np.transpose(temp_head_STOR_series)).ravel()
            temp_head_STOR_series = temp_head_STOR_series[np.isfinite(temp_head_STOR_series)]
            h_STOR_series_hourly[:,HPP] = temp_head_STOR_series
            
            # [display] once BAL simulation is complete
            print("done")
            
            
            ###############################################################
            ############----STATISTICS FOR DATA PARSING -------############
            ###############################################################
            
            # [arrange] medians in hydropower generation in GWh/year
            E_hydro_STOR_statistics_median[HPP] = 1e-3*np.median(E_hydro_STOR_yearly[:,HPP])

            # [arrange] yearly totals of HSW generation in MWh/year
            E_HSW_STOR_yearly[:,HPP] = E_hydro_STOR_flexible_yearly[:,HPP] + E_hydro_STOR_stable_yearly[:,HPP] + E_solar_STOR_yearly[:,HPP] + E_wind_STOR_yearly[:,HPP]
            
            # [arrange] medians and IQ ranges of HSW generation and ELCC in GWh/year
            E_HSW_STOR_statistics_median[HPP] = 1e-3*np.median(E_HSW_STOR_yearly[:,HPP])
            E_HSW_STOR_statistics_pct25[HPP] = 1e-3*np.percentile(E_HSW_STOR_yearly[:,HPP],25)
            E_HSW_STOR_statistics_pct75[HPP] = 1e-3*np.percentile(E_HSW_STOR_yearly[:,HPP],75)
            
            E_solar_STOR_statistics_median[HPP] = 1e-3*np.median(E_solar_STOR_yearly[:,HPP])
            E_solar_STOR_statistics_pct25[HPP] = 1e-3*np.percentile(E_solar_STOR_yearly[:,HPP],25)
            E_solar_STOR_statistics_pct75[HPP] = 1e-3*np.percentile(E_solar_STOR_yearly[:,HPP],75)
            
            E_wind_STOR_statistics_median[HPP] = 1e-3*np.median(E_wind_STOR_yearly[:,HPP])
            E_wind_STOR_statistics_pct25[HPP] = 1e-3*np.percentile(E_solar_STOR_yearly[:,HPP],25)
            E_wind_STOR_statistics_pct75[HPP] = 1e-3*np.percentile(E_solar_STOR_yearly[:,HPP],75)
            
            ELCC_STOR_statistics_median[HPP] = 1e-3*np.median(ELCC_STOR_yearly[:,HPP])
            ELCC_STOR_statistics_pct25[HPP] = 1e-3*np.percentile(ELCC_STOR_yearly[:,HPP],25)
            ELCC_STOR_statistics_pct75[HPP] = 1e-3*np.percentile(ELCC_STOR_yearly[:,HPP],75)
            
            # [arrange] ratio of yearly ELCC to yearly hydropower generation (STOR)
            ratio_ELCC_E_hydro_STOR_yearly[:,HPP] = ELCC_STOR_yearly[:,HPP]/E_hydro_STOR_yearly[:,HPP]
            ratio_ELCC_E_hydro_STOR_median[HPP] = np.nanmedian(ratio_ELCC_E_hydro_STOR_yearly[:,HPP])

            # [calculate] hourly hydropower capacity factor for BAL (eq. S42)
            CF_hydro_STOR_hourly[:,:,HPP] = (P_STOR_hydro_stable_hourly[:,:,HPP] + P_STOR_hydro_flexible_hourly[:,:,HPP])/(P_r_turb[HPP])
            
            # [calculate] turbine exhaustion factor k_turb in BAL (eq. S28)
            k_turb_hourly_STOR[:,:,HPP] = (Q_STOR_stable_hourly[:,:,HPP] + Q_STOR_flexible_hourly[:,:,HPP])/Q_max_turb[HPP]
            
            # [check] if criterion on k_turb is met for STOR, wrap up simulation and write data
            if np.median(np.nanpercentile(k_turb_hourly_STOR[:,:,HPP],99,0)) < 1:
                # [arrange] data for tables in SI B (column 7-12 in Table S3-7)
                data_SI_B_STOR[HPP,:] = np.array([P_r_turb[HPP], C_OR_range_STOR[q], E_hydro_STOR_statistics_median[HPP], 0, E_solar_STOR_statistics_median[HPP], E_wind_STOR_statistics_median[HPP], ELCC_STOR_statistics_median[HPP]])
                break
            else:
                # [display] in case k_turb criterion was not met (eq. S28)
                print("requires resimulating at lower C_OR...")
            
        
    else:
        
        c_multiplier_STOR[:,HPP] = np.nan
        E_solar_STOR_yearly[:,HPP] = np.nan
        E_wind_STOR_yearly[:,HPP] = np.nan
        E_hydro_STOR_pump_yearly[:,HPP] = np.nan
        E_hydro_STOR_flexible_yearly[:,HPP] = np.nan
        E_hydro_STOR_stable_yearly[:,HPP] = np.nan
        E_SW_STOR_yearly[:,HPP] = np.nan
        E_overproduced_STOR_yearly[:,HPP] = np.nan
        E_SW_STOR_without_overproduction[:,HPP] = np.nan
        ELCC_STOR_yearly[:,HPP] = np.nan
        
    
# [display] signal simulation end
print("simulation finished")