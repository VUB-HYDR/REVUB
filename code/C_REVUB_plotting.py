# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:43:56 2020

@author: ssterl
"""

# © 2019 CIREG project
# Author: Sebastian Sterl, Vrije Universiteit Brussel
# This code accompanies the paper "Smart renewable portfolios to displace fossil fuels and avoid hydropower overexploitation" by Sterl et al.
# All equation, section &c. numbers refer to that paper and its Supplementary Materials, unless otherwise mentioned.

import numpy as np
import pandas as pd
import numbers as nb
import matplotlib.pyplot as plt
import numpy.matlib

# [set by user] select hydropower plant and year for which to display results
plot_HPP = 1
plot_year = 14

# [set by user] select month of year (1 = Jan, 2 = Feb, &c.) and day of month, and number of days to display results
plot_month = 4
plot_day_month = 2
plot_num_days = 3

# [set by user] select months and hours of day (= o'clock) for which to show release rules
plot_rules_month = [4]
plot_rules_hr = [8, 20]

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
        months_byyear[m,y] = months_names_full[m] + np.str(simulation_years[y])

# [arrange] create string for each day-month-year combination in the time series
days_bymonth_byyear = np.empty(shape = (int(np.max(days_year)), months_yr,len(simulation_years)), dtype = 'object')
for y in range(len(simulation_years)):
    for m in range(months_yr):
        for d in range(int(days_year[m,y])):
            days_bymonth_byyear[d,m,y] = np.str(d+1) + months_names_full[m] + 'Yr' + np.str(y+1)

days_bymonth_byyear_axis = (np.transpose(days_bymonth_byyear[:,:,plot_year])).ravel()
days_bymonth_byyear_axis = list(filter(None, days_bymonth_byyear_axis))

# [colours] for plotting
colour_nat = np.array([77, 175, 74]) / 255
colour_CONV = np.array([55, 126, 184]) / 255
colour_BAL = np.array([228, 26, 28]) / 255
colour_STOR = np.array([255, 255, 51]) / 255
colour_orange = np.array([255, 127, 0]) / 255
colour_hydro_stable = np.array([55, 126, 184]) / 255
colour_hydro_flexible = np.array([106, 226, 207]) / 255
colour_solar = np.array([255, 255, 51]) / 255
colour_wind = np.array([77, 175, 74]) / 255
colour_hydro_RoR = np.array([100, 100, 100]) / 255
colour_hydro_pumped = np.array([77, 191, 237]) / 255

# [preallocate] to aggregate inflow by month
Q_in_nat_monthly_total = np.zeros(shape = (months_yr,len(simulation_years),HPP_number))

# [preallocate] to aggregate output variables by month for CONV
E_CONV_stable_bymonth = np.zeros(shape = (months_yr,len(simulation_years),HPP_number))

# [preallocate] to aggregate output variables by month for BAL
E_hydro_BAL_stable_bymonth = np.zeros(shape = (months_yr,len(simulation_years),HPP_number))
E_solar_BAL_bymonth = np.zeros(shape = (months_yr,len(simulation_years),HPP_number))
E_wind_BAL_bymonth = np.zeros(shape = (months_yr,len(simulation_years),HPP_number))
E_hydro_BAL_flexible_bymonth = np.zeros(shape = (months_yr,len(simulation_years),HPP_number))
E_hydro_BAL_RoR_bymonth = np.zeros(shape = (months_yr,len(simulation_years),HPP_number))
ELCC_BAL_byyear = np.zeros(shape = (len(simulation_years),HPP_number))
L_norm_bymonth = np.zeros(shape = (months_yr,len(simulation_years),HPP_number))
hydro_BAL_curtailment_factor_monthly = np.zeros(shape = (months_yr,len(simulation_years),HPP_number))

# [preallocate] to aggregate output variables by month for STOR
E_hydro_STOR_stable_bymonth = np.zeros(shape = (months_yr,len(simulation_years),HPP_number))
E_solar_STOR_bymonth = np.zeros(shape = (months_yr,len(simulation_years),HPP_number))
E_wind_STOR_bymonth = np.zeros(shape = (months_yr,len(simulation_years),HPP_number))
E_hydro_STOR_flexible_bymonth = np.zeros(shape = (months_yr,len(simulation_years),HPP_number))
E_hydro_pump_STOR_bymonth = np.zeros(shape = (months_yr,len(simulation_years),HPP_number))
ELCC_STOR_byyear = np.zeros(shape = (len(simulation_years),HPP_number))
hydro_STOR_curtailment_factor_monthly = np.zeros(shape = (months_yr,len(simulation_years),HPP_number))


# [loop] across all hydropower plants to aggregate output variables by month
for HPP in range(HPP_number):
    # [loop] across all years in the simulation
    for y in range(len(simulation_years)):
        # [loop] across all months of the year, converting hourly values (MW or MWh/h) to GWh/month (see eq. S24, S25)
            for m in range(months_yr):
                
                E_CONV_stable_bymonth[m,y,HPP] = 1e-3*np.sum(P_CONV_hydro_stable_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP])
                
                E_hydro_BAL_stable_bymonth[m,y,HPP] = 1e-3*np.sum(P_BAL_hydro_stable_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP])
                E_solar_BAL_bymonth[m,y,HPP] = 1e-3*np.sum(P_BAL_solar_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP])
                E_wind_BAL_bymonth[m,y,HPP] = 1e-3*np.sum(P_BAL_wind_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP])
                E_hydro_BAL_flexible_bymonth[m,y,HPP] = 1e-3*np.sum(P_BAL_hydro_flexible_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP])
                E_hydro_BAL_RoR_bymonth[m,y,HPP] = 1e-3*np.sum(P_BAL_hydro_RoR_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP])
                
                L_norm_bymonth[m,y,HPP] = np.mean(L_norm[int(np.sum(days_year[range(m),y])*hrs_day) : int(np.sum(days_year[range(m+1),y])*hrs_day),y,HPP])
                hydro_BAL_curtailment_factor_monthly[m,y,HPP] = np.min(hydro_BAL_curtailment_factor_hourly[int(np.sum(days_year[range(m),y])*hrs_day) : int(np.sum(days_year[range(m+1),y])*hrs_day),y,HPP])
                
                E_hydro_STOR_stable_bymonth[m,y,HPP] = 1e-3*np.sum(P_STOR_hydro_stable_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP])
                E_solar_STOR_bymonth[m,y,HPP] = 1e-3*np.sum(P_STOR_solar_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP])
                E_wind_STOR_bymonth[m,y,HPP] = 1e-3*np.sum(P_STOR_wind_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP])
                E_hydro_STOR_flexible_bymonth[m,y,HPP] = 1e-3*np.sum(P_STOR_hydro_flexible_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP])
                E_hydro_pump_STOR_bymonth[m,y,HPP] = 1e-3*np.sum(P_STOR_pump_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP])
                
                hydro_STOR_curtailment_factor_monthly[m,y,HPP] = np.min(hydro_STOR_curtailment_factor_hourly[int(np.sum(days_year[range(m),y])*hrs_day) : int(np.sum(days_year[range(m+1),y])*hrs_day),y,HPP])
                
                Q_in_nat_monthly_total[m,y,HPP] = np.mean(Q_in_nat_hourly[int(positions[m,y]):int(positions[m+1,y]),y,HPP])
                
            # [calculate] achieved ELCC per year for BAL (eq. S23)
            if P_followed_BAL_index[y,HPP] == 0:
                ELCC_BAL_byyear[y,HPP] = 0
            else:
                ELCC_BAL_byyear[y,HPP] = P_followed_BAL_range[y,int(P_followed_BAL_index[y,HPP]),HPP]
            
            # [calculate] achieved ELCC per year for STOR
            if P_followed_STOR_index[y,HPP] == 0:
                ELCC_STOR_byyear[y,HPP] = 0
            else:
                ELCC_STOR_byyear[y,HPP] = P_followed_STOR_range[y,int(P_followed_STOR_index[y,HPP]),HPP]
            
        
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
#plt.tight_layout()
#plt.show

# [plot] reservoir bathymetric relationship: area-volume
ax2.plot(calibrate_volume[:,plot_HPP], calibrate_area[:,plot_HPP])
plt.tight_layout()
ax2.set_xlim([0, V_max[plot_HPP]])
ax2.set_ylim([0, A_max[plot_HPP]*1.1])
ax2.set_xlabel('$V$ (m$^3$)')
ax2.set_ylabel('$A$ (m$^2$)')
ax2.set_title('area vs. volume')

plt.tight_layout()
plt.savefig("Fig1.png", dpi = 300)
plt.show


# [figure] (cf. Fig. S6)
fig = plt.figure()
ax1 = plt.subplot(221)
ax2 = plt.subplot(223)
ax3 = plt.subplot(122)

# [plot] hydraulic head time series (Fig. S6a)
ax1.plot(h_CONV_series_hourly[:,plot_HPP], color = colour_CONV)
ax1.plot(h_BAL_series_hourly[:,plot_HPP], color = colour_BAL)
if STOR_break[plot_HPP] == 0:
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
if STOR_break[plot_HPP] == 0:
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
ax3.fill_between(np.array(range(months_yr)), np.nanpercentile(Q_in_nat_monthly_total[:,:,plot_HPP], 25, axis = 1), np.nanpercentile(Q_in_nat_monthly_total[:,:,plot_HPP], 75, axis = 1), facecolor = colour_nat)
ax3.fill_between(np.array(range(months_yr)), np.nanpercentile(Q_CONV_out_monthly[:,:,plot_HPP],25, axis = 1), np.nanpercentile(Q_CONV_out_monthly[:,:,plot_HPP],75, axis = 1), facecolor = colour_CONV)
ax3.fill_between(np.array(range(months_yr)), np.nanpercentile(Q_BAL_out_monthly[:,:,plot_HPP], 25, axis = 1), np.nanpercentile(Q_BAL_out_monthly[:,:,plot_HPP], 75, axis = 1), facecolor = colour_BAL)
if STOR_break[plot_HPP] == 0:
    ax3.fill_between(np.array(range(months_yr)), np.nanpercentile(Q_STOR_out_monthly[:,:,plot_HPP], 25, axis = 1), np.nanpercentile(Q_STOR_out_monthly[:,:,plot_HPP], 75, axis = 1), facecolor = colour_STOR)
ax3.plot(np.array(range(months_yr)), np.nanpercentile(Q_in_nat_monthly_total[:,:,plot_HPP], 50, axis = 1), color = 'green')
ax3.plot(np.array(range(months_yr)), np.nanpercentile(Q_CONV_out_monthly[:,:,plot_HPP], 50, axis = 1), color = 'blue')
ax3.plot(np.array(range(months_yr)), np.nanpercentile(Q_BAL_out_monthly[:,:,plot_HPP], 50, axis = 1), color = 'red')
if STOR_break[plot_HPP] == 0:
    ax3.plot(np.array(range(months_yr)), np.nanpercentile(Q_STOR_out_monthly[:,:,plot_HPP], 50, axis = 1), color = 'orange')
    ax3.legend(['$Q_{in,nat}$', '$Q_{out,CONV}$', '$Q_{out,BAL}$', '$Q_{out,STOR}$'])
else:
    ax3.legend(['$Q_{in,nat}$', '$Q_{out,CONV}$', '$Q_{out,BAL}$'])
ax3.set_xticks(np.array(range(months_yr)))
ax3.set_xticklabels(months_names_short)
ax3.set_ylabel('$Q$ (m$^3$/s)')
ax3.set_title('inflow vs. outflow')

plt.tight_layout()
plt.savefig("Fig2.png", dpi = 300)
plt.show


# [figure]
# [plot] lake volume time series, monthly inflow vs. outflow
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(V_CONV_series_hourly[:,plot_HPP], color = colour_CONV)
ax1.plot(V_BAL_series_hourly[:,plot_HPP], color = colour_BAL)
if STOR_break[plot_HPP] == 0:
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

temp = Q_in_nat_monthly_total[:,:,plot_HPP]
ax2.plot((np.transpose(temp)).ravel(), color = colour_nat)
temp = Q_CONV_out_monthly[:,:,plot_HPP]
ax2.plot((np.transpose(temp)).ravel(), color = colour_CONV)
temp = Q_BAL_out_monthly[:,:,plot_HPP]
ax2.plot((np.transpose(temp)).ravel(), color = colour_BAL)
if STOR_break[plot_HPP] == 0:
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
plt.savefig("Fig3.png", dpi = 300)
plt.show


# [figure] (cf. Fig. S4a, S9a)
# [plot] multi-year average monthly power mix in user-selected year
fig = plt.figure()
area_mix_BAL_bymonth = [E_hydro_BAL_stable_bymonth[:,plot_year,plot_HPP], E_hydro_BAL_flexible_bymonth[:,plot_year,plot_HPP], E_wind_BAL_bymonth[:,plot_year,plot_HPP], E_solar_BAL_bymonth[:,plot_year,plot_HPP], E_hydro_BAL_RoR_bymonth[:,plot_year,plot_HPP]]/days_year[:,plot_year]*10**3/hrs_day
labels_generation_BAL = ['Hydropower (stable)', 'Hydropower (flexible)', 'Wind power', 'Solar power', 'Hydropower (RoR)']
labels_load = 'Total load'
plt.stackplot(np.array(range(months_yr)), area_mix_BAL_bymonth, labels = labels_generation_BAL, colors = [colour_hydro_stable, colour_hydro_flexible, colour_wind, colour_solar, colour_hydro_RoR])
plt.plot(np.array(range(months_yr)), L_norm_bymonth[:,plot_year,plot_HPP]*ELCC_BAL_byyear[plot_year,plot_HPP], label = labels_load, color = 'black', linewidth = 3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(np.array(range(months_yr)),months_names_full, rotation = 'vertical')
plt.ylabel('Power generation (MWh/h)')
plt.title('monthly power generation (selected year #' + str(plot_year + 1) + ', BAL)')
plt.savefig("Fig4.png", dpi = 300, bbox_inches = 'tight')


# [figure] (cf. Fig. S4b, S9b)
# [plot] power mix by year
fig = plt.figure()
E_generated_BAL_bymonth_sum = [np.sum(E_hydro_BAL_stable_bymonth[:,:,plot_HPP], axis = 0), np.sum(E_hydro_BAL_flexible_bymonth[:,:,plot_HPP], axis = 0), np.sum(E_wind_BAL_bymonth[:,:,plot_HPP], axis = 0), np.sum(E_solar_BAL_bymonth[:,:,plot_HPP], axis = 0), np.sum(E_hydro_BAL_RoR_bymonth[:,:,plot_HPP], axis = 0)]
plt.bar(np.array(range(len(simulation_years))), E_generated_BAL_bymonth_sum[0], bottom = np.sum(E_generated_BAL_bymonth_sum[0:0], axis = 0), color = colour_hydro_stable, label = 'Hydropower (stable)')
plt.bar(np.array(range(len(simulation_years))), E_generated_BAL_bymonth_sum[1], bottom = np.sum(E_generated_BAL_bymonth_sum[0:1], axis = 0), label = 'Hydropower (flexible)', color = colour_hydro_flexible)
plt.bar(np.array(range(len(simulation_years))), E_generated_BAL_bymonth_sum[2], bottom = np.sum(E_generated_BAL_bymonth_sum[0:2], axis = 0), label = 'Wind power', color = colour_wind)
plt.bar(np.array(range(len(simulation_years))), E_generated_BAL_bymonth_sum[3], bottom = np.sum(E_generated_BAL_bymonth_sum[0:3], axis = 0), label = 'Solar power', color = colour_solar)
plt.bar(np.array(range(len(simulation_years))), E_generated_BAL_bymonth_sum[4], bottom = np.sum(E_generated_BAL_bymonth_sum[0:4], axis = 0), label = 'Hydropower (RoR)', color = colour_hydro_RoR)
plt.plot(np.array(range(len(simulation_years))), ELCC_BAL_yearly[:,plot_HPP]/10**3, label = 'Total load', color = 'black', linewidth = 3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(np.array(range(len(simulation_years))), np.array(range(len(simulation_years))) + 1)
plt.xlabel('year')
plt.ylabel('Power generation (GWh/year)')
plt.ylim([0, np.max(np.sum(E_generated_BAL_bymonth_sum, axis = 0))*1.1])
plt.title('Multiannual generation (BAL)')
plt.savefig("Fig5.png", dpi = 300, bbox_inches = 'tight')


# [figure] (cf. Fig. 2 main paper, Fig. S5)
# [plot] power mix for selected days of selected month
fig = plt.figure()
area_mix_full = [P_BAL_hydro_stable_hourly[hrs_year,plot_year,plot_HPP], P_BAL_hydro_flexible_hourly[hrs_year,plot_year,plot_HPP], P_BAL_wind_hourly[hrs_year,plot_year,plot_HPP], P_BAL_solar_hourly[hrs_year,plot_year,plot_HPP], P_BAL_hydro_RoR_hourly[hrs_year,plot_year,plot_HPP]]
plt.stackplot(np.array(hrs_year), area_mix_full, labels = labels_generation_BAL, colors = [colour_hydro_stable, colour_hydro_flexible, colour_wind, colour_solar, colour_hydro_RoR])
ELCC_BAL_byday = P_followed_BAL_range[plot_year, int(P_followed_BAL_index[plot_year,plot_HPP]), plot_HPP]*L_norm[hrs_year,plot_year,plot_HPP]
plt.plot(np.array(hrs_year), ELCC_BAL_byday, label = 'Total load', color = 'black', linewidth = 3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(np.array(np.arange(hrs_year[0],hrs_year[-1] + hrs_day,hrs_day)), days_bymonth_byyear_axis)
plt.xlim([hrs_day*plot_day_load, hrs_day*(plot_day_load + plot_num_days)])
plt.ylim([0, np.max(np.sum(area_mix_full, axis = 0)*1.1)])
plt.xlabel('Day of the year')
plt.ylabel('Power generation (MWh/h)')
plt.title('Daily generation & load profiles (BAL)')
plt.savefig("Fig6.png", dpi = 300, bbox_inches = 'tight')


# [figure] if STOR scenario available
# STOR
if STOR_break[plot_HPP] == 0:
    
    # [plot] power mix by month in selected year
    fig = plt.figure()
    area_mix_STOR_bymonth = [E_hydro_STOR_stable_bymonth[:,plot_year,plot_HPP], E_hydro_STOR_flexible_bymonth[:,plot_year,plot_HPP], E_wind_STOR_bymonth[:,plot_year,plot_HPP], E_solar_STOR_bymonth[:,plot_year,plot_HPP], -1*E_hydro_pump_STOR_bymonth[:,plot_year,plot_HPP]]/days_year[:,plot_year]*10**3/hrs_day
    labels_generation_STOR = ['Hydropower (stable)', 'Hydropower (flexible)', 'Wind power', 'Solar power', 'Pump-stored excess']
    labels_load = 'Total load'
    plt.stackplot(np.array(range(months_yr)), area_mix_STOR_bymonth, labels = labels_generation_STOR, colors = [colour_hydro_stable, colour_hydro_flexible, colour_wind, colour_solar, colour_hydro_pumped])
    plt.plot(np.array(range(months_yr)), L_norm_bymonth[:,plot_year,plot_HPP]*ELCC_STOR_byyear[plot_year,plot_HPP], label = labels_load, color = 'black', linewidth = 3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(np.array(range(months_yr)),months_names_full, rotation = 'vertical')
    plt.ylabel('Power generation (MWh/h)')
    plt.title('monthly power generation (selected year #' + str(plot_year + 1) + ', STOR)')
    plt.savefig("Fig4_b.png", dpi = 300, bbox_inches = 'tight')
    
    
    # [plot] power mix by year
    fig = plt.figure()
    E_generated_STOR_bymonth_sum = [np.sum(E_hydro_STOR_stable_bymonth[:,:,plot_HPP], axis = 0), np.sum(E_hydro_STOR_flexible_bymonth[:,:,plot_HPP], axis = 0), np.sum(E_wind_STOR_bymonth[:,:,plot_HPP], axis = 0), np.sum(E_solar_STOR_bymonth[:,:,plot_HPP], axis = 0), -1*np.sum(E_hydro_pump_STOR_bymonth[:,:,plot_HPP], axis = 0)]
    plt.bar(np.array(range(len(simulation_years))), E_generated_STOR_bymonth_sum[0], bottom = np.sum(E_generated_STOR_bymonth_sum[0:0], axis = 0), color = colour_hydro_stable, label = 'Hydropower (stable)')
    plt.bar(np.array(range(len(simulation_years))), E_generated_STOR_bymonth_sum[1], bottom = np.sum(E_generated_STOR_bymonth_sum[0:1], axis = 0), label = 'Hydropower (flexible)', color = colour_hydro_flexible)
    plt.bar(np.array(range(len(simulation_years))), E_generated_STOR_bymonth_sum[2], bottom = np.sum(E_generated_STOR_bymonth_sum[0:2], axis = 0), label = 'Wind power', color = colour_wind)
    plt.bar(np.array(range(len(simulation_years))), E_generated_STOR_bymonth_sum[3], bottom = np.sum(E_generated_STOR_bymonth_sum[0:3], axis = 0), label = 'Solar power', color = colour_solar)
    plt.bar(np.array(range(len(simulation_years))), E_generated_STOR_bymonth_sum[4], bottom = np.sum(E_generated_STOR_bymonth_sum[0:4], axis = 0), label = 'Pump-stored excess', color = colour_hydro_pumped)
    plt.plot(np.array(range(len(simulation_years))), ELCC_STOR_yearly[:,plot_HPP]/10**3, label = 'Total load', color = 'black', linewidth = 3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(np.array(range(len(simulation_years))), np.array(range(len(simulation_years))) + 1)
    plt.xlabel('year')
    plt.ylabel('Power generation (GWh/year)')
    plt.ylim([0, np.max(np.sum(E_generated_BAL_bymonth_sum, axis = 0))*1.1])
    plt.title('Multiannual generation (STOR)')
    plt.savefig("Fig5_b.png", dpi = 300, bbox_inches = 'tight')
    
    
    # [plot] power mix for selected days of selected month
    fig = plt.figure()
    area_mix_full = [P_STOR_hydro_stable_hourly[hrs_year,plot_year,plot_HPP], P_STOR_hydro_flexible_hourly[hrs_year,plot_year,plot_HPP], P_STOR_wind_hourly[hrs_year,plot_year,plot_HPP], P_STOR_solar_hourly[hrs_year,plot_year,plot_HPP], -1*P_STOR_pump_hourly[hrs_year,plot_year,plot_HPP]]
    plt.stackplot(np.array(hrs_year), area_mix_full, labels = labels_generation_STOR, colors = [colour_hydro_stable, colour_hydro_flexible, colour_wind, colour_solar, colour_hydro_pumped])
    ELCC_STOR_byday = P_followed_STOR_range[plot_year, int(P_followed_STOR_index[plot_year,plot_HPP]), plot_HPP]*L_norm[hrs_year,plot_year,plot_HPP]
    plt.plot(np.array(hrs_year), ELCC_STOR_byday, label = 'Total load', color = 'black', linewidth = 3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(np.array(np.arange(hrs_year[0],hrs_year[-1] + hrs_day,hrs_day)), days_bymonth_byyear_axis)
    plt.xlim([hrs_day*plot_day_load, hrs_day*(plot_day_load + plot_num_days)])
    plt.ylim([0, np.max(np.sum(area_mix_full, axis = 0)*1.1)])
    plt.xlabel('Day of the year')
    plt.ylabel('Power generation (MWh/h)')
    plt.title('Daily generation & load profiles (STOR)')
    plt.savefig("Fig6_b.png", dpi = 300, bbox_inches = 'tight')


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
            temp_Q_025[y] = np.percentile(temp,25)
            temp_Q_075[y] = np.percentile(temp,75)
            
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
        
        plt.errorbar(temp_h, temp_Q, yerr = yerr, fmt='^', label = ('total outflow ' + str(hr) + 'h ' + str(months_names_full[m])) )
        temp_fit = np.polyfit(temp_h, temp_Q, 1)
        plt.plot(temp_h, temp_fit[1] + temp_fit[0]*temp_h, color = 'black', linestyle = '--', label = ('total outflow ' + str(hr) + 'h ' + str(months_names_full[m]) + ' fit') )
    

plt.plot([np.nanmin(h_BAL_hourly[:,:,plot_HPP]), np.nanmax(h_BAL_hourly[:,:,plot_HPP])], [np.nanmean(Q_BAL_stable_hourly[:,:,plot_HPP]), np.nanmean(Q_BAL_stable_hourly[:,:,plot_HPP])], color = 'black', label = 'stable outflow')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Hydraulic head (m)')
plt.ylabel('$Q_{out}$ (m$^3$/s)')
plt.title('release rules (BAL)')
plt.savefig("Fig7.png", dpi = 300, bbox_inches = 'tight')

