# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:43:56 2020

@author: ssterl
"""

##########################################
######### REVUB plotting results #########
##########################################

# © 2019 CIREG project
# Author: Sebastian Sterl, Vrije Universiteit Brussel
# This code accompanies the paper "Smart renewable electricity portfolios in West Africa" by Sterl et al.
# All equation, section &c. numbers refer to that paper's Supplementary Information or equivalently the REVUB manual.

import numpy as np
import pandas as pd
import numbers as nb
import matplotlib.pyplot as plt
import numpy.matlib


# import Excel file with user specifications on plotting
filename_plotting = 'plotting_settings.xlsx'

# [load] plotting parameters
parameters_plotting_multiple = pd.read_excel (filename_plotting, sheet_name = 'Plot power output (multi HPP)', header = None)
parameters_plotting_multiple_list = np.array(parameters_plotting_multiple[0][0:].tolist())
parameters_plotting_multiple_values = np.array(parameters_plotting_multiple)[0:,1:]

# [set by user] select hydropower plant (by name) and year (starting count at one) for which to display results
plot_HPP_name_multiple = parameters_plotting_multiple_values[np.where(parameters_plotting_multiple_list == 'plot_HPP_multiple', True, False)][0]
plot_HPP_multiple = np.full(len(plot_HPP_name_multiple), np.nan)
for n in range(len(plot_HPP_name_multiple)):
    temp = plot_HPP_name_multiple.astype(str)[n]
    if temp == 'nan':
        plot_HPP_multiple[n] = -1
    else:
        plot_HPP_multiple[n] = np.where(np.array(HPP_name) == temp)[0]
plot_HPP_multiple = plot_HPP_multiple.astype(int)[plot_HPP_multiple.astype(int) >= 0]

plot_year_multiple = int(parameters_plotting_multiple_values[:,0][np.where(parameters_plotting_multiple_list == 'plot_year_multiple', True, False)][0]) - 1

# [set by user] select month of year (1 = Jan, 2 = Feb, &c.) and day of month, and number of days to display results
plot_month_multiple = int(parameters_plotting_multiple_values[:,0][np.where(parameters_plotting_multiple_list == 'plot_month_multiple', True, False)][0])
plot_day_month_multiple = int(parameters_plotting_multiple_values[:,0][np.where(parameters_plotting_multiple_list == 'plot_day_month_multiple', True, False)][0])
plot_num_days_multiple = int(parameters_plotting_multiple_values[:,0][np.where(parameters_plotting_multiple_list == 'plot_num_days_multiple', True, False)][0])

# [set by user] select whether or not to plot ELCC (0 = no, 1 = yes)
plot_ELCC_line_multiple = int(parameters_plotting_multiple_values[:,0][np.where(parameters_plotting_multiple_list == 'plot_ELCC_line_multiple', True, False)][0])

# [set by user] total electricity demand to be met (MW) - these numbers are chosen for illustrative purposes only
P_total_av = int(parameters_plotting_multiple_values[:,0][np.where(parameters_plotting_multiple_list == 'P_total_av', True, False)][0])
L_norm_HPP = parameters_plotting_multiple_values[:,0][np.where(parameters_plotting_multiple_list == 'chosen_load', True, False)][0]
P_total_hourly = P_total_av*L_norm[:,:,np.where(np.array(HPP_name_data_load) == L_norm_HPP)[0][0]]

# [calculate] non-hydro-solar-wind (thermal) power contribution (difference between total and hydro-solar-wind)
P_BAL_thermal_hourly = P_total_hourly - np.nansum(P_BAL_hydro_stable_hourly[:,:,plot_HPP_multiple] + P_BAL_hydro_flexible_hourly[:,:,plot_HPP_multiple] + P_BAL_wind_hourly[:,:,plot_HPP_multiple] + P_BAL_solar_hourly[:,:,plot_HPP_multiple] + P_BAL_hydro_RoR_hourly[:,:,plot_HPP_multiple], axis = 2)
P_STOR_thermal_hourly = P_total_hourly - np.nansum(P_STOR_hydro_stable_hourly[:,:,plot_HPP_multiple] + P_STOR_hydro_flexible_hourly[:,:,plot_HPP_multiple] + P_STOR_wind_hourly[:,:,plot_HPP_multiple] + P_STOR_solar_hourly[:,:,plot_HPP_multiple] + P_BAL_hydro_RoR_hourly[:,:,plot_HPP_multiple] - P_STOR_pump_hourly[:,:,plot_HPP_multiple], axis = 2)

P_BAL_thermal_hourly[P_BAL_thermal_hourly < 0] = 0
P_STOR_thermal_hourly[P_STOR_thermal_hourly < 0] = 0

# [calculate] excess (to-be-curtailed) power
P_BAL_curtailed_hourly = np.nansum(P_BAL_hydro_stable_hourly[:,:,plot_HPP_multiple] + P_BAL_hydro_flexible_hourly[:,:,plot_HPP_multiple] + P_BAL_wind_hourly[:,:,plot_HPP_multiple] + P_BAL_solar_hourly[:,:,plot_HPP_multiple] + P_BAL_hydro_RoR_hourly[:,:,plot_HPP_multiple], axis = 2) + P_BAL_thermal_hourly - P_total_hourly
P_STOR_curtailed_hourly = np.nansum(P_STOR_hydro_stable_hourly[:,:,plot_HPP_multiple] + P_STOR_hydro_flexible_hourly[:,:,plot_HPP_multiple] + P_STOR_wind_hourly[:,:,plot_HPP_multiple] + P_STOR_solar_hourly[:,:,plot_HPP_multiple] + P_BAL_hydro_RoR_hourly[:,:,plot_HPP_multiple] - P_STOR_pump_hourly[:,:,plot_HPP_multiple], axis = 2) + P_STOR_thermal_hourly - P_total_hourly

# [preallocate] extra variables for thermal power generation assessment
E_total_bymonth = np.zeros(shape = (months_yr,len(simulation_years)))
E_thermal_BAL_bymonth = np.zeros(shape = (months_yr,len(simulation_years)))
E_thermal_STOR_bymonth = np.zeros(shape = (months_yr,len(simulation_years)))
E_curtailed_BAL_bymonth = np.zeros(shape = (months_yr,len(simulation_years)))
E_curtailed_STOR_bymonth = np.zeros(shape = (months_yr,len(simulation_years)))

# [loop] across all years in the simulation
for y in range(len(simulation_years)):
    # [loop] across all months of the year, converting hourly values (MW or MWh/h) to GWh/month (see eq. S24, S25)
    for m in range(months_yr):
        E_total_bymonth[m,y] = 10**(-3)*np.sum(P_total_hourly[int(positions[m,y]):int(positions[m+1,y]),y])
        E_thermal_BAL_bymonth[m,y] = 10**(-3)*np.sum(P_BAL_thermal_hourly[int(positions[m,y]):int(positions[m+1,y]),y])
        E_thermal_STOR_bymonth[m,y] = 10**(-3)*np.sum(P_STOR_thermal_hourly[int(positions[m,y]):int(positions[m+1,y]),y])
        E_curtailed_BAL_bymonth[m,y] = 10**(-3)*np.sum(P_BAL_curtailed_hourly[int(positions[m,y]):int(positions[m+1,y]),y])
        E_curtailed_STOR_bymonth[m,y] = 10**(-3)*np.sum(P_STOR_curtailed_hourly[int(positions[m,y]):int(positions[m+1,y]),y])
        

# [read] vector with hours in each year
hrs_year = range(int(hrs_byyear[plot_year_multiple]))

# [identify] index of day of month to plot
plot_day_load = np.sum(days_year[range(plot_month_multiple - 1),plot_year_multiple]) + plot_day_month_multiple - 1

# [strings] string arrays containing the names and abbreviations of the different months
months_names_full = np.array(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
months_names_short = np.array(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
months_byyear = np.empty(shape = (months_yr,len(simulation_years)), dtype = 'object')

# [arrange] create string for each month-year combination in the time series
for y in range(len(simulation_years)):
    for m in range(months_yr):
        months_byyear[m,y] = months_names_full[m] + str(simulation_years[y])

# [arrange] create string for each day-month-year combination in the time series
days_bymonth_byyear = np.empty(shape = (int(np.max(days_year)), months_yr,len(simulation_years)), dtype = 'object')
for y in range(len(simulation_years)):
    for m in range(months_yr):
        for d in range(int(days_year[m,y])):
            days_bymonth_byyear[d,m,y] = str(d+1) + months_names_full[m] + 'Yr' + str(y+1)

days_bymonth_byyear_axis = (np.transpose(days_bymonth_byyear[:,:,plot_year_multiple])).ravel()
days_bymonth_byyear_axis = numpy.append(days_bymonth_byyear_axis, 'NextYear')
days_bymonth_byyear_axis = list(filter(None, days_bymonth_byyear_axis))

# [colours] for plotting
colour_hydro_stable = np.array([55, 126, 184]) / 255
colour_hydro_flexible = np.array([106, 226, 207]) / 255
colour_solar = np.array([255, 255, 51]) / 255
colour_wind = np.array([77, 175, 74]) / 255
colour_hydro_RoR = np.array([158, 202, 225]) / 255
colour_hydro_pumped = np.array([77, 191, 237]) / 255
colour_thermal = np.array([75, 75, 75]) / 255
colour_curtailed = np.array([200, 200, 200]) / 255


# [figure] (cf. Fig. S4a, S9a)
# [plot] average monthly power mix in user-selected year
fig = plt.figure()
area_mix_BAL_bymonth = [np.nansum(E_hydro_BAL_stable_bymonth[:,plot_year_multiple,plot_HPP_multiple], axis = 1), np.nansum(E_hydro_BAL_flexible_bymonth[:,plot_year_multiple,plot_HPP_multiple], axis = 1), np.nansum(E_wind_BAL_bymonth[:,plot_year_multiple,plot_HPP_multiple], axis = 1), np.nansum(E_solar_BAL_bymonth[:,plot_year_multiple,plot_HPP_multiple], axis = 1), np.nansum(E_hydro_BAL_RoR_bymonth[:,plot_year_multiple,plot_HPP_multiple], axis = 1), E_thermal_BAL_bymonth[:,plot_year_multiple], -1*E_curtailed_BAL_bymonth[:,plot_year_multiple]]/days_year[:,plot_year_multiple]*10**3/hrs_day
labels_generation_BAL = ['Hydropower (stable)', 'Hydropower (flexible)', 'Wind power', 'Solar power', 'Hydropower (RoR)', 'Thermal', 'Curtailed VRE']
plt.stackplot(np.array(range(months_yr)), area_mix_BAL_bymonth, labels = labels_generation_BAL, colors = [colour_hydro_stable, colour_hydro_flexible, colour_wind, colour_solar, colour_hydro_RoR, colour_thermal, colour_curtailed])
plt.plot(np.array(range(months_yr)), E_total_bymonth[:,plot_year_multiple]/days_year[:,plot_year_multiple]*10**3/hrs_day, label = 'Total load', color = 'black', linewidth = 3)
if plot_ELCC_line_multiple == 1: plt.plot(np.array(range(months_yr)), np.nansum(ELCC_BAL_bymonth[:,plot_year_multiple,plot_HPP_multiple], axis = 1), label = 'ELCC$_{tot}$', color = 'black', linestyle = '--', linewidth = 3)
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.xticks(np.array(range(months_yr)), months_names_full, rotation = 'vertical')
plt.ylabel('Power generation (MWh/h)')
plt.title('Monthly power generation (selected year #' + str(plot_year_multiple + 1) + ', BAL)')
plt.savefig("Total_Fig1.png", dpi = 300, bbox_inches = 'tight')


# [figure] (cf. Fig. S4b, S9b)
# [plot] power mix by year
fig = plt.figure()
E_generated_BAL_bymonth_sum = [np.nansum(np.sum(E_hydro_BAL_stable_bymonth[:,:,plot_HPP_multiple], axis = 0), axis = 1), np.nansum(np.sum(E_hydro_BAL_flexible_bymonth[:,:,plot_HPP_multiple], axis = 0), axis = 1), np.nansum(np.sum(E_wind_BAL_bymonth[:,:,plot_HPP_multiple], axis = 0), axis = 1), np.nansum(np.sum(E_solar_BAL_bymonth[:,:,plot_HPP_multiple], axis = 0), axis = 1), np.nansum(np.sum(E_hydro_BAL_RoR_bymonth[:,:,plot_HPP_multiple], axis = 0), axis = 1), np.sum(E_thermal_BAL_bymonth, axis = 0), -1*np.sum(E_curtailed_BAL_bymonth, axis = 0)]
plt.bar(np.array(range(len(simulation_years))), E_generated_BAL_bymonth_sum[0], bottom = np.sum(E_generated_BAL_bymonth_sum[0:0], axis = 0), label = 'Hydropower (stable)', color = colour_hydro_stable)
plt.bar(np.array(range(len(simulation_years))), E_generated_BAL_bymonth_sum[1], bottom = np.sum(E_generated_BAL_bymonth_sum[0:1], axis = 0), label = 'Hydropower (flexible)', color = colour_hydro_flexible)
plt.bar(np.array(range(len(simulation_years))), E_generated_BAL_bymonth_sum[2], bottom = np.sum(E_generated_BAL_bymonth_sum[0:2], axis = 0), label = 'Wind power', color = colour_wind)
plt.bar(np.array(range(len(simulation_years))), E_generated_BAL_bymonth_sum[3], bottom = np.sum(E_generated_BAL_bymonth_sum[0:3], axis = 0), label = 'Solar power', color = colour_solar)
plt.bar(np.array(range(len(simulation_years))), E_generated_BAL_bymonth_sum[4], bottom = np.sum(E_generated_BAL_bymonth_sum[0:4], axis = 0), label = 'Hydropower (RoR)', color = colour_hydro_RoR)
plt.bar(np.array(range(len(simulation_years))), E_generated_BAL_bymonth_sum[5], bottom = np.sum(E_generated_BAL_bymonth_sum[0:5], axis = 0), label = 'Thermal', color = colour_thermal)
plt.bar(np.array(range(len(simulation_years))), E_generated_BAL_bymonth_sum[6], bottom = np.sum(E_generated_BAL_bymonth_sum[0:6], axis = 0), label = 'Curtailed VRE', color = colour_curtailed)
plt.plot(np.array(range(len(simulation_years))), np.sum(E_total_bymonth, axis = 0), label = 'Total load', color = 'black', linewidth = 3)
if plot_ELCC_line_multiple == 1: plt.plot(np.array(range(len(simulation_years))), np.sum(ELCC_BAL_yearly[:,plot_HPP_multiple], axis = 1)/10**3, label = 'ELCC$_{tot}$', color = 'black', linestyle = '--', linewidth = 3)
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.xticks(np.array(range(len(simulation_years))), np.array(range(len(simulation_years))) + 1)
plt.xlabel('year')
plt.ylabel('Power generation (GWh/year)')
plt.ylim([0, np.nanmax(np.sum(E_generated_BAL_bymonth_sum, axis = 0))*1.1])
plt.title('Multiannual generation (BAL)')
plt.savefig("Total_Fig2.png", dpi = 300, bbox_inches = 'tight')


# [figure] (cf. Fig. 2 main paper, Fig. S5)
# [plot] power mix for selected days of selected month
fig = plt.figure()
area_mix_full = [np.nansum(P_BAL_hydro_stable_hourly[hrs_year,plot_year_multiple,plot_HPP_multiple[:,np.newaxis]], axis = 0), np.nansum(P_BAL_hydro_flexible_hourly[hrs_year,plot_year_multiple,plot_HPP_multiple[:,np.newaxis]], axis = 0), np.nansum(P_BAL_wind_hourly[hrs_year,plot_year_multiple,plot_HPP_multiple[:,np.newaxis]], axis = 0), np.nansum(P_BAL_solar_hourly[hrs_year,plot_year_multiple,plot_HPP_multiple[:,np.newaxis]], axis = 0), np.nansum(P_BAL_hydro_RoR_hourly[hrs_year,plot_year_multiple,plot_HPP_multiple[:,np.newaxis]], axis = 0), P_BAL_thermal_hourly[hrs_year,plot_year_multiple], -1*P_BAL_curtailed_hourly[hrs_year,plot_year_multiple]]
plt.stackplot(np.array(hrs_year), area_mix_full, labels = labels_generation_BAL, colors = [colour_hydro_stable, colour_hydro_flexible, colour_wind, colour_solar, colour_hydro_RoR, colour_thermal, colour_curtailed])
plt.plot(np.array(hrs_year), P_total_hourly[hrs_year,plot_year_multiple], label = 'Total load', color = 'black', linewidth = 3)
if plot_ELCC_line_multiple == 1: plt.plot(np.array(hrs_year), np.nansum(L_followed_BAL_hourly[hrs_year,plot_year_multiple,plot_HPP_multiple[:,np.newaxis]], axis = 0), label = 'ELCC$_{tot}$', color = 'black', linestyle = '--', linewidth = 3)
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.xticks(np.array(np.arange(hrs_year[0],hrs_year[-1] + hrs_day,hrs_day)), days_bymonth_byyear_axis)
plt.xlim([hrs_day*plot_day_load, hrs_day*(plot_day_load + plot_num_days_multiple)])
plt.ylim([0, np.nanmax(np.sum(area_mix_full, axis = 0)*1.1)])
plt.xlabel('Day of the year')
plt.ylabel('Power generation (MWh/h)')
plt.title('Daily generation & load profiles (BAL)')
plt.savefig("Total_Fig3.png", dpi = 300, bbox_inches = 'tight')


# [check] if STOR scenario available
if option_storage == 1 and np.min(STOR_break[plot_HPP_multiple]) == 0:
    
    # [figure] (cf. Fig. S4a, S9a)
    # [plot] average monthly power mix in user-selected year
    fig = plt.figure()
    area_mix_STOR_bymonth = [np.nansum(E_hydro_STOR_stable_bymonth[:,plot_year_multiple,plot_HPP_multiple], axis = 1), np.nansum(E_hydro_STOR_flexible_bymonth[:,plot_year_multiple,plot_HPP_multiple], axis = 1), np.nansum(E_wind_STOR_bymonth[:,plot_year_multiple,plot_HPP_multiple], axis = 1), np.nansum(E_solar_STOR_bymonth[:,plot_year_multiple,plot_HPP_multiple] - E_hydro_pump_STOR_bymonth[:,plot_year_multiple,plot_HPP_multiple], axis = 1), np.nansum(E_hydro_BAL_RoR_bymonth[:,plot_year_multiple,plot_HPP_multiple], axis = 1), E_thermal_STOR_bymonth[:,plot_year_multiple], -1*E_curtailed_STOR_bymonth[:,plot_year_multiple]]/days_year[:,plot_year_multiple]*10**3/hrs_day
    labels_generation_STOR = ['Hydropower (stable)', 'Hydropower (flexible)', 'Wind power', 'Solar power', 'Hydropower (RoR)', 'Thermal', 'Curtailed VRE']
    plt.stackplot(np.array(range(months_yr)), area_mix_STOR_bymonth, labels = labels_generation_STOR, colors = [colour_hydro_stable, colour_hydro_flexible, colour_wind, colour_solar, colour_hydro_RoR, colour_thermal, colour_curtailed])
    plt.fill_between(np.array(range(months_yr)), -1*np.nansum(E_hydro_pump_STOR_bymonth[:,plot_year_multiple,plot_HPP_multiple], axis = 1), label = 'Stored VRE', facecolor = colour_hydro_pumped)
    plt.plot(np.array(range(months_yr)), E_total_bymonth[:,plot_year_multiple]/days_year[:,plot_year_multiple]*10**3/hrs_day, label = 'Total load', color = 'black', linewidth = 3)
    plt.plot(np.array(range(months_yr)), np.nansum(ELCC_STOR_bymonth[:,plot_year_multiple,plot_HPP_multiple], axis = 1), label = 'ELCC$_{tot}$', color = 'black', linestyle = '--', linewidth = 3)
    plt.plot(np.array(range(months_yr)), np.zeros(months_yr), color = 'black', linewidth = 1)
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.xticks(np.array(range(months_yr)),months_names_full, rotation = 'vertical')
    plt.ylabel('Power generation (MWh/h)')
    plt.title('monthly power generation (selected year #' + str(plot_year_multiple + 1) + ', STOR)')
    plt.savefig("Total_Fig1_b.png", dpi = 300, bbox_inches = 'tight')
    
    
    # [figure] (cf. Fig. S4b, S9b)
    # [plot] power mix by year
    fig = plt.figure()
    E_generated_STOR_bymonth_sum = [np.nansum(np.sum(E_hydro_STOR_stable_bymonth[:,:,plot_HPP_multiple], axis = 0), axis = 1), np.nansum(np.sum(E_hydro_STOR_flexible_bymonth[:,:,plot_HPP_multiple], axis = 0), axis = 1), np.nansum(np.sum(E_wind_STOR_bymonth[:,:,plot_HPP_multiple], axis = 0), axis = 1), np.nansum(np.sum(E_solar_STOR_bymonth[:,:,plot_HPP_multiple] - E_hydro_pump_STOR_bymonth[:,:,plot_HPP_multiple], axis = 0), axis = 1), np.nansum(np.sum(E_hydro_BAL_RoR_bymonth[:,:,plot_HPP_multiple], axis = 0), axis = 1), np.sum(E_thermal_STOR_bymonth, axis = 0), -1*np.sum(E_curtailed_STOR_bymonth, axis = 0)]
    plt.bar(np.array(range(len(simulation_years))), E_generated_STOR_bymonth_sum[0], bottom = np.sum(E_generated_STOR_bymonth_sum[0:0], axis = 0), label = 'Hydropower (stable)', color = colour_hydro_stable)
    plt.bar(np.array(range(len(simulation_years))), E_generated_STOR_bymonth_sum[1], bottom = np.sum(E_generated_STOR_bymonth_sum[0:1], axis = 0), label = 'Hydropower (flexible)', color = colour_hydro_flexible)
    plt.bar(np.array(range(len(simulation_years))), E_generated_STOR_bymonth_sum[2], bottom = np.sum(E_generated_STOR_bymonth_sum[0:2], axis = 0), label = 'Wind power', color = colour_wind)
    plt.bar(np.array(range(len(simulation_years))), E_generated_STOR_bymonth_sum[3], bottom = np.sum(E_generated_STOR_bymonth_sum[0:3], axis = 0), label = 'Solar power', color = colour_solar)
    plt.bar(np.array(range(len(simulation_years))), E_generated_STOR_bymonth_sum[4], bottom = np.sum(E_generated_STOR_bymonth_sum[0:4], axis = 0), label = 'Hydropower (RoR)', color = colour_hydro_RoR)
    plt.bar(np.array(range(len(simulation_years))), E_generated_STOR_bymonth_sum[5], bottom = np.sum(E_generated_STOR_bymonth_sum[0:5], axis = 0), label = 'Thermal', color = colour_thermal)
    plt.bar(np.array(range(len(simulation_years))), E_generated_STOR_bymonth_sum[6], bottom = np.sum(E_generated_STOR_bymonth_sum[0:6], axis = 0), label = 'Curtailed VRE', color = colour_curtailed)
    plt.bar(np.array(range(len(simulation_years))), -1*np.nansum(np.sum(E_hydro_pump_STOR_bymonth[:,:,plot_HPP_multiple], axis = 0), axis = 1), label = 'Stored VRE', color = colour_hydro_pumped)
    plt.plot(np.array(range(len(simulation_years))), np.sum(E_total_bymonth, axis = 0), label = 'Total load', color = 'black', linewidth = 3)
    plt.plot(np.array(range(len(simulation_years))), np.sum(ELCC_STOR_yearly[:,plot_HPP_multiple], axis = 1)/10**3, label = 'ELCC$_{tot}$', color = 'black', linestyle = '--', linewidth = 3)
    plt.plot(np.array(range(len(simulation_years))), np.zeros(len(simulation_years)), color = 'black', linewidth = 1)
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.xticks(np.array(range(len(simulation_years))), np.array(range(len(simulation_years))) + 1)
    plt.xlabel('year')
    plt.ylabel('Power generation (GWh/year)')
    plt.ylim([np.nanmin(-1*np.sum(E_hydro_pump_STOR_bymonth[:,:,plot_HPP_multiple], axis = 0))*1.1, np.nanmax(np.sum(E_generated_STOR_bymonth_sum, axis = 0))*1.1])
    plt.title('Multiannual generation (STOR)')
    plt.savefig("Total_Fig2_b.png", dpi = 300, bbox_inches = 'tight')
    
    
    # [figure] (cf. Fig. 2 main paper, Fig. S5)
    # [plot] power mix for selected days of selected month
    fig = plt.figure()
    area_mix_full = [np.nansum(P_STOR_hydro_stable_hourly[hrs_year,plot_year_multiple,plot_HPP_multiple[:,np.newaxis]], axis = 0), np.nansum(P_STOR_hydro_flexible_hourly[hrs_year,plot_year_multiple,plot_HPP_multiple[:,np.newaxis]], axis = 0), np.nansum(P_STOR_wind_hourly[hrs_year,plot_year_multiple,plot_HPP_multiple[:,np.newaxis]], axis = 0), np.nansum(P_STOR_solar_hourly[hrs_year,plot_year_multiple,plot_HPP_multiple[:,np.newaxis]] - P_STOR_pump_hourly[hrs_year,plot_year_multiple,plot_HPP_multiple[:,np.newaxis]], axis = 0), np.nansum(P_BAL_hydro_RoR_hourly[hrs_year,plot_year_multiple,plot_HPP_multiple[:,np.newaxis]], axis = 0), P_STOR_thermal_hourly[hrs_year,plot_year_multiple], -1*P_STOR_curtailed_hourly[hrs_year,plot_year_multiple]]
    plt.stackplot(np.array(hrs_year), area_mix_full, labels = labels_generation_STOR, colors = [colour_hydro_stable, colour_hydro_flexible, colour_wind, colour_solar, colour_hydro_RoR, colour_thermal, colour_curtailed])
    plt.fill_between(np.array(hrs_year), -1*np.nansum(P_STOR_pump_hourly[hrs_year,plot_year_multiple,plot_HPP_multiple[:,np.newaxis]], axis = 0), label = 'Stored VRE', color = colour_hydro_pumped)
    plt.plot(np.array(hrs_year), P_total_hourly[hrs_year,plot_year_multiple], label = 'Total load', color = 'black', linewidth = 3)
    plt.plot(np.array(hrs_year), np.nansum(L_followed_STOR_hourly[hrs_year,plot_year_multiple,plot_HPP_multiple[:,np.newaxis]], axis = 0), label = 'ELCC$_{tot}$', color = 'black', linestyle = '--', linewidth = 3)
    plt.plot(np.array(hrs_year), np.zeros(len(hrs_year)), color = 'black', linewidth = 1)
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.xticks(np.array(np.arange(hrs_year[0],hrs_year[-1] + hrs_day,hrs_day)), days_bymonth_byyear_axis)
    plt.xlim([hrs_day*plot_day_load, hrs_day*(plot_day_load + plot_num_days_multiple)])
    plt.ylim([np.nanmin(-1*np.nansum(P_STOR_pump_hourly[hrs_year,plot_year_multiple,plot_HPP_multiple[:,np.newaxis]], axis = 0))*1.1, np.nanmax(np.sum(area_mix_full, axis = 0)*1.1)])
    plt.xlabel('Day of the year')
    plt.ylabel('Power generation (MWh/h)')
    plt.title('Daily generation & load profiles (STOR)')
    plt.savefig("Total_Fig3_b.png", dpi = 300, bbox_inches = 'tight')
