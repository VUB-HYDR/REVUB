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
parameters_plotting_single = pd.read_excel (filename_plotting, sheet_name = 'Plot power output (single HPP)', header = None)
parameters_plotting_single_list = np.array(parameters_plotting_single[0][0:].tolist())
parameters_plotting_single_values = np.array(parameters_plotting_single[1][0:].tolist())

parameters_plotting_release = pd.read_excel (filename_plotting, sheet_name = 'Plot release rules (single HPP)', header = None)
parameters_plotting_release_list = np.array(parameters_plotting_release[0][0:].tolist())
parameters_plotting_release_values = np.array(parameters_plotting_release)[0:,2:]

# [set by user] select hydropower plant (by name) and year (starting count at one) for which to display results
plot_HPP_name = parameters_plotting_single_values[np.where(parameters_plotting_single_list == 'plot_HPP', True, False)][0]
plot_HPP = np.where(np.array(HPP_name) == plot_HPP_name)[0][0]
plot_year = int(parameters_plotting_single_values[np.where(parameters_plotting_single_list == 'plot_year', True, False)][0]) - 1

# [set by user] select month of year (1 = Jan, 2 = Feb, &c.) and day of month, and number of days to display results
plot_month = int(parameters_plotting_single_values[np.where(parameters_plotting_single_list == 'plot_month', True, False)][0])
plot_day_month = int(parameters_plotting_single_values[np.where(parameters_plotting_single_list == 'plot_day_month', True, False)][0])
plot_num_days = int(parameters_plotting_single_values[np.where(parameters_plotting_single_list == 'plot_num_days', True, False)][0])

# [set by user] select months and hours of day (= o'clock) for which to show release rules
plot_rules_month = parameters_plotting_release_values[np.where(parameters_plotting_release_list == 'plot_rules_month', True, False)][0]
plot_rules_month = np.array(plot_rules_month, dtype = float)
plot_rules_month = plot_rules_month[~np.isnan(plot_rules_month)].astype(int)

plot_rules_hr = parameters_plotting_release_values[np.where(parameters_plotting_release_list == 'plot_rules_hr', True, False)][0]
plot_rules_hr = np.array(plot_rules_hr, dtype = float)
plot_rules_hr = plot_rules_hr[~np.isnan(plot_rules_hr)].astype(int)

# [read] vector with hours in each year
hrs_year = range(int(hrs_byyear[plot_year]))

# [identify] index of day of month to plot
plot_day_load = np.sum(days_year[range(plot_month - 1),plot_year]) + plot_day_month - 1

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

days_bymonth_byyear_axis = (np.transpose(days_bymonth_byyear[:,:,plot_year])).ravel()
days_bymonth_byyear_axis = numpy.append(days_bymonth_byyear_axis, 'NextYear')
days_bymonth_byyear_axis = list(filter(None, days_bymonth_byyear_axis))

# [colours] for plotting
colour_nat = np.array([77, 175, 74]) / 255
colour_CONV = np.array([55, 126, 184]) / 255
colour_BAL = np.array([228, 26, 28]) / 255
colour_STOR = np.array([255, 255, 51]) / 255
colour_hydro_stable = np.array([55, 126, 184]) / 255
colour_hydro_flexible = np.array([106, 226, 207]) / 255
colour_solar = np.array([255, 255, 51]) / 255
colour_wind = np.array([77, 175, 74]) / 255
colour_hydro_RoR = np.array([158, 202, 225]) / 255
colour_hydro_pumped = np.array([77, 191, 237]) / 255


# [calculate] take Fourier transform of CONV head time series (cf. Fig. S6b)
fft_rep = 1
fft_temp_CONV = np.transpose(np.matlib.repmat(h_CONV_series_hourly[:,plot_HPP],fft_rep,1))
fft_CONV = np.fft.fft(fft_temp_CONV, axis = 0)
L_fft_CONV = len(fft_CONV)
# [calculate] take absolute value
fft_CONV_amp = np.abs(fft_CONV/L_fft_CONV)
# [calculate] compute single-sided spectrum and normalise
fft_CONV_amp = fft_CONV_amp[range(int(L_fft_CONV/2) + 1)]
fft_CONV_amp[1:-2] = 2*fft_CONV_amp[1:-2]
fft_CONV_amp[0] = 0
fft_CONV_amp = fft_CONV_amp/np.max(fft_CONV_amp)
# [calculate] corresponding frequency series (in Hz)
fft_CONV_freq = (1/secs_hr)*np.array(range(int(L_fft_CONV/2) + 1))/L_fft_CONV

# [calculate] take Fourier transform of BAL head time series (cf. Fig. S6b)
fft_temp_BAL = np.transpose(np.matlib.repmat(h_BAL_series_hourly[:,plot_HPP],fft_rep,1))
fft_BAL = np.fft.fft(fft_temp_BAL, axis = 0)
L_fft_BAL = len(fft_BAL)
# [calculate] take absolute value
fft_BAL_amp = np.abs(fft_BAL/L_fft_BAL)
# [calculate] compute single-sided spectrum and normalise
fft_BAL_amp = fft_BAL_amp[range(int(L_fft_BAL/2) + 1)]
fft_BAL_amp[1:-2] = 2*fft_BAL_amp[1:-2]
fft_BAL_amp[0] = 0
fft_BAL_amp = fft_BAL_amp/np.max(fft_BAL_amp)
# [calculate] corresponding frequency series (in Hz)
fft_BAL_freq = (1/secs_hr)*np.array(range(int(L_fft_BAL/2) + 1))/L_fft_BAL

# [calculate] take Fourier transform of STOR head time series (cf. Fig. S6b)
fft_temp_STOR = np.transpose(np.matlib.repmat(h_STOR_series_hourly[:,plot_HPP],fft_rep,1))
fft_STOR = np.fft.fft(fft_temp_STOR, axis = 0)
L_fft_STOR = len(fft_STOR)
# [calculate] take absolute value
fft_STOR_amp = np.abs(fft_STOR/L_fft_STOR)
# [calculate] compute single-sided spectrum and normalise
fft_STOR_amp = fft_STOR_amp[range(int(L_fft_STOR/2) + 1)]
fft_STOR_amp[1:-2] = 2*fft_STOR_amp[1:-2]
fft_STOR_amp[0] = 0
fft_STOR_amp = fft_STOR_amp/np.max(fft_STOR_amp)
# [calculate] corresponding frequency series (in Hz)
fft_STOR_freq = (1/secs_hr)*np.array(range(int(L_fft_STOR/2) + 1))/L_fft_STOR

# [display] name of hydropower plant for which results are plotted
print(HPP_name[plot_HPP])


# [figure] (cf. Fig. S7)
# [plot] reservoir bathymetric relationship: head-volume
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(calibrate_volume[:,plot_HPP], calibrate_head[:,plot_HPP])
ax1.set_xlim([0, V_max[plot_HPP]])
ax1.set_ylim([0, h_max[plot_HPP]*1.1])
ax1.set_xlabel('$V$ (m$^3$)')
ax1.set_ylabel('$h$ (m)')
ax1.set_title('head vs. volume')

# [plot] reservoir bathymetric relationship: area-volume
ax2.plot(calibrate_volume[:,plot_HPP], calibrate_area[:,plot_HPP])
plt.tight_layout()
ax2.set_xlim([0, V_max[plot_HPP]])
ax2.set_ylim([0, A_max[plot_HPP]*1.1])
ax2.set_xlabel('$V$ (m$^3$)')
ax2.set_ylabel('$A$ (m$^2$)')
ax2.set_title('area vs. volume')

plt.tight_layout()
plt.savefig(HPP_name[plot_HPP] + '_Fig1.png', dpi = 300)
plt.show


# [figure] (cf. Fig. S6)
fig = plt.figure()
ax1 = plt.subplot(221)
ax2 = plt.subplot(223)
ax3 = plt.subplot(122)

# [plot] hydraulic head time series (Fig. S6a)
ax1.plot(h_CONV_series_hourly[:,plot_HPP], color = colour_CONV)
ax1.plot(h_BAL_series_hourly[:,plot_HPP], color = colour_BAL)
if STOR_break[plot_HPP] == 0 and option_storage == 1:
    ax1.plot(h_STOR_series_hourly[:,plot_HPP], color = colour_STOR, linestyle = '--')
    ax1.legend(['CONV', 'BAL', 'STOR'])
else:
    ax1.legend(['CONV', 'BAL'])
ax1.set_xticks(np.append(np.cumsum(positions[-1,:]) - positions[-1,0], np.sum(positions[-1,:])))
ax1.set_xticklabels(np.array(range(1, year_end - year_start + 3)))
ax1.set_xlabel('time (years)')
ax1.set_ylim([0, h_max[plot_HPP]])
ax1.set_ylabel('$h(t)$ (m)')
ax1.set_title('time series of hydraulic head')
plt.setp(ax1.get_xticklabels(), rotation = 90, horizontalalignment = 'right', fontsize = 'x-small')

# [plot] hydraulic head frequency response (Fig. S6b)
ax2.loglog(fft_CONV_freq, fft_CONV_amp, color = colour_CONV)
ax2.loglog(fft_BAL_freq, fft_BAL_amp, color = colour_BAL)
if STOR_break[plot_HPP] == 0 and option_storage == 1:
    ax2.loglog(fft_STOR_freq, fft_STOR_amp, color = colour_STOR, linestyle = '--')
    ax2.legend(['CONV', 'BAL', 'STOR'])
else:
    ax2.legend(['CONV', 'BAL'])
ax2.set_xlim([fft_CONV_freq[1], 1/(2*secs_hr)])
ax2.set_ylim([1e-7, 2])
ax2.set_xlabel('$f$ (s$^{-1}$)')
ax2.set_ylabel('F[$h(t)$]')
ax2.set_title('frequency spectrum of hydraulic head')

# [plot] median + IQ range inflow vs. outflow (Fig. S6c)
ax3.fill_between(np.array(range(months_yr)), np.nanpercentile(Q_in_nat_monthly[:,:,plot_HPP], 25, axis = 1), np.nanpercentile(Q_in_nat_monthly[:,:,plot_HPP], 75, axis = 1), facecolor = colour_nat)
ax3.fill_between(np.array(range(months_yr)), np.nanpercentile(Q_CONV_out_monthly[:,:,plot_HPP],25, axis = 1), np.nanpercentile(Q_CONV_out_monthly[:,:,plot_HPP],75, axis = 1), facecolor = colour_CONV)
ax3.fill_between(np.array(range(months_yr)), np.nanpercentile(Q_BAL_out_monthly[:,:,plot_HPP], 25, axis = 1), np.nanpercentile(Q_BAL_out_monthly[:,:,plot_HPP], 75, axis = 1), facecolor = colour_BAL)
if STOR_break[plot_HPP] == 0 and option_storage == 1:
    ax3.fill_between(np.array(range(months_yr)), np.nanpercentile(Q_STOR_out_monthly[:,:,plot_HPP], 25, axis = 1), np.nanpercentile(Q_STOR_out_monthly[:,:,plot_HPP], 75, axis = 1), facecolor = colour_STOR)
ax3.plot(np.array(range(months_yr)), np.nanpercentile(Q_in_nat_monthly[:,:,plot_HPP], 50, axis = 1), color = 'green')
ax3.plot(np.array(range(months_yr)), np.nanpercentile(Q_CONV_out_monthly[:,:,plot_HPP], 50, axis = 1), color = 'blue')
ax3.plot(np.array(range(months_yr)), np.nanpercentile(Q_BAL_out_monthly[:,:,plot_HPP], 50, axis = 1), color = 'red')
if STOR_break[plot_HPP] == 0 and option_storage == 1:
    ax3.plot(np.array(range(months_yr)), np.nanpercentile(Q_STOR_out_monthly[:,:,plot_HPP], 50, axis = 1), color = 'orange')
    ax3.legend(['$Q_{in,nat}$', '$Q_{out,CONV}$', '$Q_{out,BAL}$', '$Q_{out,STOR}$'])
else:
    ax3.legend(['$Q_{in,nat}$', '$Q_{out,CONV}$', '$Q_{out,BAL}$'])
ax3.set_xticks(np.array(range(months_yr)))
ax3.set_xticklabels(months_names_short)
ax3.set_ylabel('$Q$ (m$^3$/s)')
ax3.set_title('inflow vs. outflow')

plt.tight_layout()
plt.savefig(HPP_name[plot_HPP] + '_Fig2.png', dpi = 300)
plt.show


# [figure]
# [plot] lake volume time series, monthly inflow vs. outflow
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(V_CONV_series_hourly[:,plot_HPP], color = colour_CONV)
ax1.plot(V_BAL_series_hourly[:,plot_HPP], color = colour_BAL)
if STOR_break[plot_HPP] == 0 and option_storage == 1:
    ax1.plot(V_STOR_series_hourly_upper[:,plot_HPP], color = colour_STOR)
    ax1.legend(['CONV', 'BAL', 'STOR'])
else:
    ax1.legend(['CONV', 'BAL'])
ax1.set_xticks(np.append(np.cumsum(positions[-1,:]) - positions[-1,0], np.sum(positions[-1,:])))
ax1.set_xticklabels(np.array(range(1, year_end - year_start + 3)))
ax1.set_ylim([0, V_max[plot_HPP]])
ax1.set_xlabel('time (years)')
ax1.set_ylabel('$V(t)$ (m$^3$)')
ax1.set_title('time series of lake volume')

temp = Q_in_nat_monthly[:,:,plot_HPP]
ax2.plot((np.transpose(temp)).ravel(), color = colour_nat)
temp = Q_CONV_out_monthly[:,:,plot_HPP]
ax2.plot((np.transpose(temp)).ravel(), color = colour_CONV)
temp = Q_BAL_out_monthly[:,:,plot_HPP]
ax2.plot((np.transpose(temp)).ravel(), color = colour_BAL)
if STOR_break[plot_HPP] == 0 and option_storage == 1:
    temp = Q_STOR_out_monthly[:,:,plot_HPP]
    ax2.plot((np.transpose(temp)).ravel(), color = colour_STOR)
    ax2.legend(['$Q_{in}$', '$Q_{out,CONV}$', '$Q_{out,BAL}$', '$Q_{out,STOR}$'])
else:
    ax2.legend(['$Q_{in}$', '$Q_{out,CONV}$', '$Q_{out,BAL}$'])
ax2.set_xticks(np.arange(0, months_yr*(len(simulation_years) + 1), months_yr))
ax2.set_xticklabels(np.array(range(1, year_end - year_start + 3)))
ax2.set_xlabel('time (years)')
ax2.set_ylabel('$Q(t)$ (m$^3$/s)')
ax2.set_title('inflow vs. outflow (monthly)')

plt.tight_layout()
plt.savefig(HPP_name[plot_HPP] + '_Fig3.png', dpi = 300)
plt.show


# [plot] BAL and STOR scenario results, in case plant deemed suitable for flexibility
if d_min[plot_HPP] != 1:

    # [figure] (cf. Fig. S4a, S9a)
    # [plot] average monthly power mix in user-selected year
    fig = plt.figure()
    area_mix_BAL_bymonth = [E_hydro_BAL_stable_bymonth[:,plot_year,plot_HPP], E_hydro_BAL_flexible_bymonth[:,plot_year,plot_HPP], E_wind_BAL_bymonth[:,plot_year,plot_HPP], E_solar_BAL_bymonth[:,plot_year,plot_HPP], E_hydro_BAL_RoR_bymonth[:,plot_year,plot_HPP]]/days_year[:,plot_year]*10**3/hrs_day
    labels_generation_BAL = ['Hydropower (stable)', 'Hydropower (flexible)', 'Wind power', 'Solar power', 'Hydropower (RoR)']
    labels_load = 'ELCC'
    plt.stackplot(np.array(range(months_yr)), area_mix_BAL_bymonth, labels = labels_generation_BAL, colors = [colour_hydro_stable, colour_hydro_flexible, colour_wind, colour_solar, colour_hydro_RoR])
    plt.plot(np.array(range(months_yr)), ELCC_BAL_bymonth[:,plot_year,plot_HPP], label = labels_load, color = 'black', linewidth = 3)
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.xticks(np.array(range(months_yr)),months_names_full, rotation = 'vertical')
    plt.ylabel('Power generation (MWh/h)')
    plt.title('monthly power generation (selected year #' + str(plot_year + 1) + ', BAL)')
    plt.savefig(HPP_name[plot_HPP] + '_Fig4.png', dpi = 300, bbox_inches = 'tight')
    
    
    # [figure] (cf. Fig. S4b, S9b)
    # [plot] power mix by year
    fig = plt.figure()
    E_generated_BAL_bymonth_sum = [np.sum(E_hydro_BAL_stable_bymonth[:,:,plot_HPP], axis = 0), np.sum(E_hydro_BAL_flexible_bymonth[:,:,plot_HPP], axis = 0), np.sum(E_wind_BAL_bymonth[:,:,plot_HPP], axis = 0), np.sum(E_solar_BAL_bymonth[:,:,plot_HPP], axis = 0), np.sum(E_hydro_BAL_RoR_bymonth[:,:,plot_HPP], axis = 0)]
    plt.bar(np.array(range(len(simulation_years))), E_generated_BAL_bymonth_sum[0], bottom = np.sum(E_generated_BAL_bymonth_sum[0:0], axis = 0), label = 'Hydropower (stable)', color = colour_hydro_stable)
    plt.bar(np.array(range(len(simulation_years))), E_generated_BAL_bymonth_sum[1], bottom = np.sum(E_generated_BAL_bymonth_sum[0:1], axis = 0), label = 'Hydropower (flexible)', color = colour_hydro_flexible)
    plt.bar(np.array(range(len(simulation_years))), E_generated_BAL_bymonth_sum[2], bottom = np.sum(E_generated_BAL_bymonth_sum[0:2], axis = 0), label = 'Wind power', color = colour_wind)
    plt.bar(np.array(range(len(simulation_years))), E_generated_BAL_bymonth_sum[3], bottom = np.sum(E_generated_BAL_bymonth_sum[0:3], axis = 0), label = 'Solar power', color = colour_solar)
    plt.bar(np.array(range(len(simulation_years))), E_generated_BAL_bymonth_sum[4], bottom = np.sum(E_generated_BAL_bymonth_sum[0:4], axis = 0), label = 'Hydropower (RoR)', color = colour_hydro_RoR)
    plt.plot(np.array(range(len(simulation_years))), ELCC_BAL_yearly[:,plot_HPP]/10**3, label = 'ELCC', color = 'black', linewidth = 3)
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.xticks(np.array(range(len(simulation_years))), np.array(range(len(simulation_years))) + 1)
    plt.xlabel('year')
    plt.ylabel('Power generation (GWh/year)')
    plt.ylim([0, np.max(np.sum(E_generated_BAL_bymonth_sum, axis = 0))*1.1])
    plt.title('Multiannual generation (BAL)')
    plt.savefig(HPP_name[plot_HPP] + '_Fig5.png', dpi = 300, bbox_inches = 'tight')
    
    
    # [figure] (cf. Fig. 2 main paper, Fig. S5)
    # [plot] power mix for selected days of selected month
    fig = plt.figure()
    area_mix_full = [P_BAL_hydro_stable_hourly[hrs_year,plot_year,plot_HPP], P_BAL_hydro_flexible_hourly[hrs_year,plot_year,plot_HPP], P_BAL_wind_hourly[hrs_year,plot_year,plot_HPP], P_BAL_solar_hourly[hrs_year,plot_year,plot_HPP], P_BAL_hydro_RoR_hourly[hrs_year,plot_year,plot_HPP]]
    plt.stackplot(np.array(hrs_year), area_mix_full, labels = labels_generation_BAL, colors = [colour_hydro_stable, colour_hydro_flexible, colour_wind, colour_solar, colour_hydro_RoR])
    plt.plot(np.array(hrs_year), L_followed_BAL_hourly[hrs_year,plot_year,plot_HPP], label = 'ELCC', color = 'black', linewidth = 3)
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.xticks(np.array(np.arange(hrs_year[0],hrs_year[-1] + hrs_day,hrs_day)), days_bymonth_byyear_axis)
    plt.xlim([hrs_day*plot_day_load, hrs_day*(plot_day_load + plot_num_days)])
    plt.ylim([0, np.max(np.sum(area_mix_full, axis = 0)*1.1)])
    plt.xlabel('Day of the year')
    plt.ylabel('Power generation (MWh/h)')
    plt.title('Daily generation & load profiles (BAL)')
    plt.savefig(HPP_name[plot_HPP] + '_Fig6.png', dpi = 300, bbox_inches = 'tight')
    
    
    # [check] if STOR scenario available
    if STOR_break[plot_HPP] == 0 and option_storage == 1:
        
        # [figure] (cf. Fig. S4a, S9a)
        # [plot] average monthly power mix in user-selected year
        fig = plt.figure()
        area_mix_STOR_bymonth = [E_hydro_STOR_stable_bymonth[:,plot_year,plot_HPP], E_hydro_STOR_flexible_bymonth[:,plot_year,plot_HPP], E_wind_STOR_bymonth[:,plot_year,plot_HPP], E_solar_STOR_bymonth[:,plot_year,plot_HPP], -1*E_hydro_pump_STOR_bymonth[:,plot_year,plot_HPP]]/days_year[:,plot_year]*10**3/hrs_day
        labels_generation_STOR = ['Hydropower (stable)', 'Hydropower (flexible)', 'Wind power', 'Solar power', 'Stored VRE']
        labels_load = 'ELCC'
        plt.stackplot(np.array(range(months_yr)), area_mix_STOR_bymonth, labels = labels_generation_STOR, colors = [colour_hydro_stable, colour_hydro_flexible, colour_wind, colour_solar, colour_hydro_pumped])
        plt.plot(np.array(range(months_yr)), ELCC_STOR_bymonth[:,plot_year,plot_HPP], label = labels_load, color = 'black', linewidth = 3)
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        plt.xticks(np.array(range(months_yr)),months_names_full, rotation = 'vertical')
        plt.ylabel('Power generation (MWh/h)')
        plt.title('monthly power generation (selected year #' + str(plot_year + 1) + ', STOR)')
        plt.savefig(HPP_name[plot_HPP] + '_Fig4_b.png', dpi = 300, bbox_inches = 'tight')
        
        
        # [figure] (cf. Fig. S4b, S9b)
        # [plot] power mix by year
        fig = plt.figure()
        E_generated_STOR_bymonth_sum = [np.sum(E_hydro_STOR_stable_bymonth[:,:,plot_HPP], axis = 0), np.sum(E_hydro_STOR_flexible_bymonth[:,:,plot_HPP], axis = 0), np.sum(E_wind_STOR_bymonth[:,:,plot_HPP], axis = 0), np.sum(E_solar_STOR_bymonth[:,:,plot_HPP], axis = 0), -1*np.sum(E_hydro_pump_STOR_bymonth[:,:,plot_HPP], axis = 0)]
        plt.bar(np.array(range(len(simulation_years))), E_generated_STOR_bymonth_sum[0], bottom = np.sum(E_generated_STOR_bymonth_sum[0:0], axis = 0), label = 'Hydropower (stable)', color = colour_hydro_stable)
        plt.bar(np.array(range(len(simulation_years))), E_generated_STOR_bymonth_sum[1], bottom = np.sum(E_generated_STOR_bymonth_sum[0:1], axis = 0), label = 'Hydropower (flexible)', color = colour_hydro_flexible)
        plt.bar(np.array(range(len(simulation_years))), E_generated_STOR_bymonth_sum[2], bottom = np.sum(E_generated_STOR_bymonth_sum[0:2], axis = 0), label = 'Wind power', color = colour_wind)
        plt.bar(np.array(range(len(simulation_years))), E_generated_STOR_bymonth_sum[3], bottom = np.sum(E_generated_STOR_bymonth_sum[0:3], axis = 0), label = 'Solar power', color = colour_solar)
        plt.bar(np.array(range(len(simulation_years))), E_generated_STOR_bymonth_sum[4], bottom = np.sum(E_generated_STOR_bymonth_sum[0:4], axis = 0), label = 'Stored VRE', color = colour_hydro_pumped)
        plt.plot(np.array(range(len(simulation_years))), ELCC_STOR_yearly[:,plot_HPP]/10**3, label = 'ELCC', color = 'black', linewidth = 3)
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        plt.xticks(np.array(range(len(simulation_years))), np.array(range(len(simulation_years))) + 1)
        plt.xlabel('year')
        plt.ylabel('Power generation (GWh/year)')
        plt.ylim([0, np.max(np.sum(E_generated_STOR_bymonth_sum, axis = 0))*1.1])
        plt.title('Multiannual generation (STOR)')
        plt.savefig(HPP_name[plot_HPP] + '_Fig5_b.png', dpi = 300, bbox_inches = 'tight')
        
        
        # [figure] (cf. Fig. 2 main paper, Fig. S5)
        # [plot] power mix for selected days of selected month
        fig = plt.figure()
        area_mix_full = [P_STOR_hydro_stable_hourly[hrs_year,plot_year,plot_HPP], P_STOR_hydro_flexible_hourly[hrs_year,plot_year,plot_HPP], P_STOR_wind_hourly[hrs_year,plot_year,plot_HPP], P_STOR_solar_hourly[hrs_year,plot_year,plot_HPP], -1*P_STOR_pump_hourly[hrs_year,plot_year,plot_HPP]]
        plt.stackplot(np.array(hrs_year), area_mix_full, labels = labels_generation_STOR, colors = [colour_hydro_stable, colour_hydro_flexible, colour_wind, colour_solar, colour_hydro_pumped])
        plt.plot(np.array(hrs_year), L_followed_STOR_hourly[hrs_year,plot_year,plot_HPP], label = 'ELCC', color = 'black', linewidth = 3)
        plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        plt.xticks(np.array(np.arange(hrs_year[0],hrs_year[-1] + hrs_day,hrs_day)), days_bymonth_byyear_axis)
        plt.xlim([hrs_day*plot_day_load, hrs_day*(plot_day_load + plot_num_days)])
        plt.ylim([0, np.max(np.sum(area_mix_full, axis = 0)*1.1)])
        plt.xlabel('Day of the year')
        plt.ylabel('Power generation (MWh/h)')
        plt.title('Daily generation & load profiles (STOR)')
        plt.savefig(HPP_name[plot_HPP] + '_Fig6_b.png', dpi = 300, bbox_inches = 'tight')
    
    
    # [preallocate] vectors for adapted rules under BAL
    h_BAL_rules_stable_bymonth = np.zeros(shape = (months_yr,len(simulation_years)))
    h_BAL_rules_total_bymonth = np.zeros(shape = (months_yr,len(simulation_years),int(np.max(days_year)),hrs_day))
    Q_out_net_BAL_rules_stable_bymonth = np.full([months_yr,len(simulation_years)], np.nan)
    Q_out_net_BAL_rules_total_bymonth = np.full([months_yr,len(simulation_years),int(np.max(days_year)),hrs_day], np.nan)
    
    # [loop] across all simulation years to calculate release for each day, hour, month and year
    for y in range(len(simulation_years)):
        # [loop] across all months in each year
        for m in range(months_yr):
            # [arrange] hourly head and outflow values for each month
            temp_head_BAL_bymonth = h_BAL_hourly[int(positions[m,y]):int(positions[m+1,y]),y,plot_HPP]
            temp_Q_BAL_bymonth = Q_BAL_out_hourly[int(positions[m,y]):int(positions[m+1,y]),y,plot_HPP] - Q_in_RoR_hourly[int(positions[m,y]):int(positions[m+1,y]),y,plot_HPP] - Q_BAL_spill_hourly[int(positions[m,y]):int(positions[m+1,y]),y,plot_HPP]
            temp_Q_BAL_bymonth[temp_Q_BAL_bymonth < 0] = np.nan
            
            # [arrange] according to specific hours of day
            # [loop] across all hours of the day
            for hr in range(hrs_day):
                
                # [find] head during specific hours
                temp_head_bymonth_hr = temp_head_BAL_bymonth[hr::hrs_day]
                
                # [find] outflow during specific hours
                temp_Q_bymonth_hr = temp_Q_BAL_bymonth[hr::hrs_day]
                
                # [loop] across all days of the month to find rules for flexible outflow
                for day in range(int(days_year[m,y])):
                    
                    # [arrange] head and outflow for each hour, day, month, year
                    h_BAL_rules_total_bymonth[m,y,day,hr] = temp_head_bymonth_hr[day]
                    Q_out_net_BAL_rules_total_bymonth[m,y,day,hr] = temp_Q_bymonth_hr[day]
                
            
        
    # [figure] approximate release rules for selected months and hours of day (Fig. 2b main paper)
    fig = plt.figure()
    
    # [loop] across selected hours of day
    for hr in plot_rules_hr:
        # [loop] across selected months
        for m in np.array(plot_rules_month) - 1:
            
            # [preallocate] temporary vectors to store head and outflow
            temp_h = np.zeros(shape = (len(simulation_years)))
            temp_Q = np.zeros(shape = (len(simulation_years)))
            temp_Q_std = np.zeros(shape = (len(simulation_years)))
            temp_Q_025 = np.zeros(shape = (len(simulation_years)))
            temp_Q_075 = np.zeros(shape = (len(simulation_years)))
            
            # [loop] across all simulation years
            for y in range(len(simulation_years)):
                
                # [calculate] head at given hour of day for all days in a single month in a single year
                temp = h_BAL_rules_total_bymonth[m,y,0:int(days_year[m,y]),hr]
                
                # [calculate] take the mean for that time of day in that month
                temp_h[y] = np.nanmedian(temp)
                
                # [calculate] outflow at given hour of day for all days in a single month in a single year
                temp = Q_out_net_BAL_rules_total_bymonth[m,y,0:int(days_year[m,y]),hr]
                
                temp_Q[y] = np.nanmedian(temp)
                temp_Q_std[y] = np.nanstd(temp)
                temp_Q_025[y] = np.nanpercentile(temp,25)
                temp_Q_075[y] = np.nanpercentile(temp,75)
                
                # [check] mark drought incidences
                if hydro_BAL_curtailment_factor_monthly[m,y,plot_HPP] == 0:
                    temp_Q[y] = np.nan
                    temp_h[y] = np.nan
                    temp_Q_std[y] = np.nan
                    temp_Q_025[y] = np.nan
                    temp_Q_075[y] = np.nan
                    
                
            # [check] remove drought incidences
            temp_Q = temp_Q[np.isfinite(temp_Q)]
            temp_h = temp_h[np.isfinite(temp_h)]
            temp_Q_std = temp_Q_std[np.isfinite(temp_Q_std)]
            temp_Q_025 = temp_Q_025[np.isfinite(temp_Q_025)]
            temp_Q_075 = temp_Q_075[np.isfinite(temp_Q_075)]
            
            # specify errorbar values
            yerr = [temp_Q - temp_Q_025, temp_Q_075 - temp_Q]
            
            plt.errorbar(temp_h, temp_Q, yerr = yerr, fmt='^', label = ('regulated outflow ' + str(hr) + 'h ' + str(months_names_full[m])) )
            temp_fit = np.polyfit(temp_h, temp_Q, 1)
            plt.plot(temp_h, temp_fit[1] + temp_fit[0]*temp_h, color = 'black', linestyle = '--', label = ('regulated outflow ' + str(hr) + 'h ' + str(months_names_full[m]) + ' fit') )
        
    
    plt.plot([np.nanmin(h_BAL_hourly[:,:,plot_HPP]), np.nanmax(h_BAL_hourly[:,:,plot_HPP])], [np.nanmean(Q_BAL_stable_hourly[:,:,plot_HPP]), np.nanmean(Q_BAL_stable_hourly[:,:,plot_HPP])], color = 'black', label = 'fixed outflow')
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.xlabel('Hydraulic head (m)')
    plt.ylabel('$Q_{out}$ (m$^3$/s)')
    plt.title('release rules (BAL)')
    plt.savefig(HPP_name[plot_HPP] + '_Fig7.png', dpi = 300, bbox_inches = 'tight')
    
    
    # [figure] plot statistics on number of turbines in use
    # [calculate] capacity of each turbine (MW)
    P_unit = P_r_turb/no_turbines
    
    # [calculate] total hydropower generation (MW) per HPP
    P_CONV_hydro_total = P_CONV_hydro_stable_hourly + P_CONV_hydro_RoR_hourly
    P_BAL_hydro_total = P_BAL_hydro_stable_hourly + P_BAL_hydro_flexible_hourly + P_BAL_hydro_RoR_hourly
    P_STOR_hydro_total = P_STOR_hydro_stable_hourly + P_STOR_hydro_flexible_hourly
    
    # [preallocate] number of turbines in use
    no_turbines_used_CONV = np.full([int(np.max(sum(days_year)))*hrs_day*len(simulation_years)], np.nan)
    no_turbines_used_BAL = np.full([int(np.max(sum(days_year)))*hrs_day*len(simulation_years)], np.nan)
    no_turbines_used_STOR = np.full([int(np.max(sum(days_year)))*hrs_day*len(simulation_years)], np.nan)
    
    # [calculate] number of turbines in use for each time step by HPP
    # [loop] across the number of turbines
    for n in range(no_turbines[plot_HPP]):
        
        # [calculate] index counting backwards from all, all-but-one, all-but two turbines, &c.
        m = no_turbines[plot_HPP] - n
        
        # [calculate] instances with all, all-but-one, all-but-two turbines active, &c.
        no_turbines_used_CONV[P_CONV_hydro_total[:,:,plot_HPP].ravel() <= m*P_unit[plot_HPP]] = m
        no_turbines_used_BAL[P_BAL_hydro_total[:,:,plot_HPP].ravel() <= m*P_unit[plot_HPP]] = m
        if STOR_break[plot_HPP] == 0:
            no_turbines_used_STOR[P_STOR_hydro_total[:,:,plot_HPP].ravel() <= m*P_unit[plot_HPP]] = m
    
    # [create] histogram of turbine use for selected HPP
    no_turbines_used_pdf_CONV = np.histogram(no_turbines_used_CONV, bins = np.array(range(1,no_turbines[plot_HPP] + 2)))
    no_turbines_used_pdf_CONV = (no_turbines_used_pdf_CONV[0])/np.sum(no_turbines_used_pdf_CONV[0])
    no_turbines_used_pdf_BAL = np.histogram(no_turbines_used_BAL, bins = np.array(range(1,no_turbines[plot_HPP] + 2)))
    no_turbines_used_pdf_BAL = (no_turbines_used_pdf_BAL[0])/np.sum(no_turbines_used_pdf_BAL[0])
    if STOR_break[plot_HPP] == 0:
        no_turbines_used_pdf_STOR = np.histogram(no_turbines_used_STOR, bins = np.array(range(1,no_turbines[plot_HPP] + 2)))
        no_turbines_used_pdf_STOR = (no_turbines_used_pdf_STOR[0])/np.sum(no_turbines_used_pdf_STOR[0])
    
    # [plot] histogram of turbine use for selected HPP
    fig = plt.figure()
    plt.bar(np.array(range(no_turbines[plot_HPP])),no_turbines_used_pdf_CONV, label = 'CONV', width = 0.8, color = colour_CONV)
    plt.bar(np.array(range(no_turbines[plot_HPP])),no_turbines_used_pdf_BAL, label = 'BAL', width = 0.6, color = colour_BAL)
    if STOR_break[plot_HPP] == 0:
        plt.bar(np.array(range(no_turbines[plot_HPP])),no_turbines_used_pdf_STOR, label = 'STOR', width = 0.4, color = colour_STOR)
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.xticks(np.array(range(no_turbines[plot_HPP])), np.array(range(no_turbines[plot_HPP])) + 1)
    plt.xlabel('Number of active turbines')
    plt.ylabel('Fraction of time')
    plt.title('Hydroturbine activity')
    plt.savefig(HPP_name[plot_HPP] + '_Fig8.png', dpi = 300, bbox_inches = 'tight')
    
    
    # [plot] histogram of flexibility regime by month in BAL regime
    hist_maxed_out_monthly = [np.count_nonzero(temp_maxed_out_BAL_monthly[:,:,plot_HPP] == 0), np.count_nonzero(temp_maxed_out_BAL_monthly[:,:,plot_HPP] == 1), np.count_nonzero(temp_maxed_out_BAL_monthly[:,:,plot_HPP] == 0.5), np.count_nonzero(temp_maxed_out_BAL_monthly[:,:,plot_HPP] == -1)]
    hist_maxed_out_monthly = hist_maxed_out_monthly/np.sum(hist_maxed_out_monthly)
    fig = plt.figure()
    plt.bar(np.array(range(len(hist_maxed_out_monthly))), hist_maxed_out_monthly, width = 0.8)
    labels_maxed_out = ['Flexibility', 'Baseload', 'Mixed', 'Curtailed']
    plt.xticks(np.array(range(len(hist_maxed_out_monthly))), labels_maxed_out)
    plt.ylabel('Fraction of time')
    plt.title('Operational regime (BAL)')
    plt.savefig(HPP_name[plot_HPP] + '_Fig9.png', dpi = 300, bbox_inches = 'tight')