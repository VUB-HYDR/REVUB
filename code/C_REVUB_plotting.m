%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% REVUB plotting results %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% © 2019 CIREG project
% Author: Sebastian Sterl, Vrije Universiteit Brussel
% This code accompanies the paper "Streamlining hydro, solar and wind energy strategies in West Africa from power plant to power pool" by Sterl et al.
% All equation, section &c. numbers refer to that paper and its Supplementary Materials, unless otherwise mentioned.

set(groot, 'defaultTextInterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
set(0,'DefaultAxesFontSize',8)
set(0,'DefaultLineLineWidth',1)


% [initialize] close windows from previous plotting
close all

% [set by user] select hydropower plant and year, month, days for which to display results
plot_HPP = 1;
plot_year = 15;
plot_month = 4;
plot_day_month = 2;
plot_num_days = 3;

% [set by user] select months and hours of day for which to show release rules
plot_rules_month = 4;
plot_rules_hr = [9 21];

% [read] vector with hours in each year
hrs_year = 1:hrs_byyear(plot_year);

% [identify] index of day of month to plot
plot_day_load = sum(days_year(1:plot_month - 1,plot_year)) + plot_day_month - 1;

% [strings] string arrays containing the names and abbreviations of the different months
months_names_full = ["Jan"; "Feb"; "Mar"; "Apr"; "May"; "Jun"; "Jul"; "Aug"; "Sep"; "Oct"; "Nov"; "Dec"; ];
months_names_short = ["J"; "F"; "M"; "A"; "M"; "J"; "J"; "A"; "S"; "O"; "N"; "D"; ];
months_byyear = strings(months_yr,length(simulation_years));

% [arrange] create string for each month-year combination in the time series
for y = 1:length(simulation_years)
    for m = 1:months_yr
        months_byyear(m,y) = strcat(months_names_full(m),num2str(simulation_years(y)));
    end
end

% [arrange] create string for each day-month-year combination in the time series
str = strings;
for y = 1:length(simulation_years)
    for m = 1:months_yr
        for d = 1:days_year(m,y)
            str(d,m,y) = strcat(num2str(d), months_names_full(m), num2str(simulation_years(y)));
        end
    end
end
str_axis = str(:,:,plot_year);
str_axis(ismissing(str_axis)) = [];

% [arrange] number of days per month as 1D array
temp = days_year(:);
% [prealocate] cumulative number of hours since start of simulation period
N_hours_cumulative = zeros(1,size(days_year,1)*size(days_year,2));
% [loop] over all days in the simulation to calculate cumulative number of hours
for n = 1:size(days_year,1)*size(days_year,2)
    N_hours_cumulative(n) = sum(temp(1:n))*hrs_day;
end
clear temp

% [colours] for plotting
colour_nat = [77, 175, 74] / 255;
colour_CONV = [55, 126, 184] / 255;
colour_BAL = [228, 26, 28] / 255;
colour_STOR = [255, 255, 51] / 255;
colour_orange = [255, 127, 0] / 255;

colour_hydro_stable = [55, 126, 184] / 255;
colour_hydro_flexible = [106 226 207] / 255;
colour_solar = [255, 255, 51] / 255;
colour_wind = [77, 175, 74] / 255;
colour_hydro_RoR = [100, 100, 100] / 255;
colour_hydro_pumped = [77, 191, 237] / 255;

% [preallocate] to aggregate inflow by month
Q_in_nat_monthly_total = zeros(months_yr,length(simulation_years),HPP_number);

% [preallocate] to aggregate output variables by month for CONV
E_CONV_stable_bymonth = zeros(months_yr,length(simulation_years),HPP_number);

% [preallocate] to aggregate output variables by month for BAL
E_hydro_BAL_stable_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
E_solar_BAL_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
E_wind_BAL_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
E_hydro_BAL_flexible_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
E_hydro_BAL_RoR_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
ELCC_BAL_byyear = zeros(length(simulation_years),HPP_number);
L_norm_BAL_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
hydro_BAL_curtailment_factor_monthly = zeros(months_yr,length(simulation_years),HPP_number); 

% [preallocate] to aggregate output variables by month for STOR
E_hydro_STOR_stable_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
E_solar_STOR_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
E_wind_STOR_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
E_hydro_STOR_flexible_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
E_hydro_pump_STOR_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
ELCC_STOR_byyear = zeros(length(simulation_years),HPP_number);
L_norm_STOR_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
hydro_STOR_curtailment_factor_monthly = zeros(months_yr,length(simulation_years),HPP_number);

% [loop] across all hydropower plants to aggregate output variables by month
for HPP = 1:HPP_number
    
    % [loop] across all years in the simulation
    for y = 1:length(simulation_years)
        
        % [loop] across all months of the year, converting hourly values (MW or MWh/h) to GWh/month (see eq. S24, S25)
        for m = 1:months_yr
            E_CONV_stable_bymonth(m,y,HPP) = 1e-3.*sum(P_CONV_hydro_stable_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            
            E_hydro_BAL_stable_bymonth(m,y,HPP) = 1e-3.*sum(P_BAL_hydro_stable_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            E_solar_BAL_bymonth(m,y,HPP) = 1e-3.*sum(P_BAL_solar_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            E_wind_BAL_bymonth(m,y,HPP) = 1e-3.*sum(P_BAL_wind_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            E_hydro_BAL_flexible_bymonth(m,y,HPP) = 1e-3.*sum(P_BAL_hydro_flexible_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            E_hydro_BAL_RoR_bymonth(m,y,HPP) = 1e-3.*sum(P_BAL_hydro_RoR_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            L_norm_BAL_bymonth(m,y,HPP) = mean(L_norm(sum(days_year(1:m-1,1))*hrs_day + 1 : sum(days_year(1:m,1))*hrs_day,y,HPP));
            hydro_BAL_curtailment_factor_monthly(m,y,HPP) = min(hydro_BAL_curtailment_factor_hourly(sum(days_year(1:m-1,1))*hrs_day + 1 : sum(days_year(1:m,1))*hrs_day,y,HPP));
            
            E_hydro_STOR_stable_bymonth(m,y,HPP) = 1e-3.*sum(P_STOR_hydro_stable_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            E_solar_STOR_bymonth(m,y,HPP) = 1e-3.*sum(P_STOR_solar_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            E_wind_STOR_bymonth(m,y,HPP) = 1e-3.*sum(P_STOR_wind_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            E_hydro_STOR_flexible_bymonth(m,y,HPP) = 1e-3.*sum(P_STOR_hydro_flexible_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            E_hydro_pump_STOR_bymonth(m,y,HPP) = 1e-3.*sum(P_STOR_pump_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            L_norm_STOR_bymonth(m,y,HPP) = mean(L_norm(sum(days_year(1:m-1,1))*hrs_day + 1 : sum(days_year(1:m,1))*hrs_day,y,HPP));
            hydro_STOR_curtailment_factor_monthly(m,y,HPP) = min(hydro_STOR_curtailment_factor_hourly(sum(days_year(1:m-1,1))*hrs_day + 1 : sum(days_year(1:m,1))*hrs_day,y,HPP));
            
            Q_in_nat_monthly_total(m,y,HPP) = mean(Q_in_nat_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
        end
        
        % [calculate] achieved ELCC per year for BAL (eq. S23)
        if P_followed_BAL_index(y,HPP) == 0
            ELCC_BAL_byyear(y,HPP) = 0;
        else
            ELCC_BAL_byyear(y,HPP) = P_followed_BAL_range(y,P_followed_BAL_index(y,HPP),HPP);
        end
        
        % [calculate] achieved ELCC per year for STOR
        if P_followed_STOR_index(y,HPP) == 0
            ELCC_STOR_byyear(y,HPP) = 0;
        else
            ELCC_STOR_byyear(y,HPP) = P_followed_STOR_range(y,P_followed_STOR_index(y,HPP),HPP);
        end
        
    end
    
end

% [calculate] take Fourier transform of CONV head time series (cf. Fig. S6b)
fft_rep = 1;
fft_temp_CONV = repmat(h_CONV_series_hourly(:,plot_HPP),fft_rep,1);
[fft_CONV] = fft(fft_temp_CONV);
L_fft_CONV = length(fft_temp_CONV);

% [calculate] take absolute value
fft_CONV_amp = abs(fft_CONV/L_fft_CONV);

% [calculate] compute single-sided spectrum and normalise
fft_CONV_amp = fft_CONV_amp(1:L_fft_CONV/2 + 1);
fft_CONV_amp(2:end-1) = 2*fft_CONV_amp(2:end-1);
fft_CONV_amp(1) = 0;
fft_CONV_amp = fft_CONV_amp./max(fft_CONV_amp);

% [calculate] corresponding frequency series (in Hz)
fft_CONV_freq = (1/secs_hr)*(0:L_fft_CONV/2)./L_fft_CONV;


% [calculate] take Fourier transform of BAL head time series (cf. Fig. S6b)
fft_temp_BAL = repmat(h_BAL_series_hourly(:,plot_HPP),fft_rep,1);
[fft_BAL] = fft(fft_temp_BAL);
L_fft_BAL = length(fft_temp_BAL);

% [calculate] take absolute value
fft_BAL_amp = abs(fft_BAL/L_fft_BAL);

% [calculate] compute single-sided spectrum and normalise
fft_BAL_amp = fft_BAL_amp(1:L_fft_BAL/2 + 1);
fft_BAL_amp(2:end-1) = 2*fft_BAL_amp(2:end-1);
fft_BAL_amp(1) = 0;
fft_BAL_amp = fft_BAL_amp./max(fft_BAL_amp);

% [calculate] corresponding frequency series
fft_BAL_freq = (1/secs_hr)*(0:L_fft_BAL/2)./L_fft_BAL;


% [calculate] take Fourier transform of STOR head time series
fft_temp_STOR = repmat(h_STOR_series_hourly(:,plot_HPP),fft_rep,1);
[fft_STOR] = fft(fft_temp_STOR);
L_fft_STOR = length(fft_temp_STOR);

% [calculate] take absolute value
fft_STOR_amp = abs(fft_STOR/L_fft_STOR);

% [calculate] compute single-sided spectrum and normalise
fft_STOR_amp = fft_STOR_amp(1:L_fft_STOR/2 + 1);
fft_STOR_amp(2:end-1) = 2*fft_STOR_amp(2:end-1);
fft_STOR_amp(1) = 0;
fft_STOR_amp = fft_STOR_amp./max(fft_STOR_amp);

% [calculate] corresponding frequency series
fft_STOR_freq = (1/secs_hr)*(0:L_fft_STOR/2)./L_fft_STOR;

% [display] name of hydropower plant for which results are plotted
HPP_name(plot_HPP)


% [figure] (cf. Fig. S7)
% [plot] reservoir bathymetric relationship: head-volume
figure()
subplot(1,2,1)
plot(calibrate_volume(:,plot_HPP),calibrate_head(:,plot_HPP))
xlim([0 V_max(plot_HPP)])
xlabel '$V$ (m$^3$)'
ylabel '$h$ (m)'
title 'head-volume relationship'

% [plot] reservoir bathymetric relationship: area-volume
subplot(1,2,2)
plot(calibrate_volume(:,plot_HPP),calibrate_area(:,plot_HPP))
xlim([0 V_max(plot_HPP)])
xlabel '$V$ (m$^3$)'
ylabel '$A$ (m$^2$)'
title 'area-volume relationship'

% [figure] (cf. Fig. S6)
% [plot] hydraulic head time series (Fig. S6a)
figure()
subplot(2,2,1)
plot(h_CONV_series_hourly(:,plot_HPP),'Color',colour_CONV)
hold on
plot(h_BAL_series_hourly(:,plot_HPP),'Color',colour_BAL)
if STOR_break(plot_HPP) == 0
    plot(h_STOR_series_hourly(:,plot_HPP),'Color',colour_STOR,'LineStyle','--')
    legend 'CONV' 'BAL' 'STOR'
else
    legend 'CONV' 'BAL'
end
xticks(N_hours_cumulative - N_hours_cumulative(1))
xticklabels(months_byyear(:))
xtickangle(90)
ylim([0 h_max(plot_HPP)])
xlabel 'time'
ylabel '$h(t)$ (m)'
title 'time series of hydraulic head'

% [plot] hydraulic head frequency response (Fig. S6b)
subplot(2,2,3)
loglog(fft_CONV_freq,fft_CONV_amp,'Color',colour_CONV)
hold on
loglog(fft_BAL_freq,fft_BAL_amp,'Color',colour_BAL)
if STOR_break(plot_HPP) == 0
    loglog(fft_STOR_freq,fft_STOR_amp,'Color',colour_STOR,'LineStyle','--')
    legend 'CONV' 'BAL' 'STOR'
else
    legend 'CONV' 'BAL'
end
xlim([fft_CONV_freq(2) 1/(2*secs_hr)])
ylim([1e-7 2])
set(gca, 'YScale', 'log')
xlabel '$f$ (s$^{-1}$)'
ylabel '$F[h(t)]$'
title 'frequency spectrum of hydraulic head'

% [plot] median + IQ range inflow vs. outflow (Fig. S6c)
subplot(2,2,[2 4])
hold on
patch([1:12 fliplr(1:12)], [prctile(Q_in_nat_monthly_total(:,:,plot_HPP)',25) fliplr(prctile(Q_in_nat_monthly_total(:,:,plot_HPP)',75))],colour_nat,'EdgeColor',colour_nat);
patch([1:12 fliplr(1:12)], [prctile(Q_CONV_out_monthly(:,:,plot_HPP)',25) fliplr(prctile(Q_CONV_out_monthly(:,:,plot_HPP)',75))],colour_CONV,'EdgeColor',colour_CONV);
patch([1:12 fliplr(1:12)], [prctile(Q_BAL_out_monthly(:,:,plot_HPP)',25) fliplr(prctile(Q_BAL_out_monthly(:,:,plot_HPP)',75))],colour_BAL,'EdgeColor',colour_BAL);
if STOR_break(plot_HPP) == 0
    patch([1:12 fliplr(1:12)], [prctile(Q_STOR_out_monthly(:,:,plot_HPP)',25) fliplr(prctile(Q_STOR_out_monthly(:,:,plot_HPP)',75))],colour_STOR,'EdgeColor',colour_STOR);
end
plot(1:12,prctile(Q_in_nat_monthly_total(:,:,plot_HPP)',50),'g--')
plot(1:12,prctile(Q_CONV_out_monthly(:,:,plot_HPP)',50),'b--')
plot(1:12,prctile(Q_BAL_out_monthly(:,:,plot_HPP)',50),'r--')
if STOR_break(plot_HPP) == 0
    plot(1:12,prctile(Q_STOR_out_monthly(:,:,plot_HPP)',50),'Color',colour_orange,'LineStyle','--')
    legend '$Q_{in,nat}$' '$Q_{out,CONV}$' '$Q_{out,BAL}$' '$Q_{out,STOR}$'
else
    legend '$Q_{in,nat}$' '$Q_{out,CONV}$' '$Q_{out,BAL}$'
end
xticks(1:months_yr)
xlim([1 months_yr])
xticklabels(months_names_short)
ylabel '$Q$ (m$^3$/s)'
title 'inflow vs. outflow (monthly median + IQ range)'

% [figure]
% [plot] lake volume time series, monthly inflow vs. outflow
figure()
subplot(2,1,1)
plot(V_CONV_series_hourly(:,plot_HPP),'Color',colour_CONV)
hold on
plot(V_BAL_series_hourly(:,plot_HPP),'Color',colour_BAL)
if STOR_break(plot_HPP) == 0
    plot(V_STOR_series_hourly_upper(:,plot_HPP),'Color',colour_STOR,'LineStyle','--')
    legend 'CONV' 'BAL' 'STOR'
else
    legend 'CONV' 'BAL'
end
xticks(N_hours_cumulative - N_hours_cumulative(1))
xticklabels(months_byyear(:))
xtickangle(90)
ylim([0 V_max(plot_HPP)])
xlabel 'time'
ylabel '$V(t)$ (m$^3$)'
title 'time series of lake volume'

subplot(2,1,2)
temp = Q_in_nat_monthly_total(:,:,plot_HPP);
plot(1:length(temp(:)),temp(:),'Color',colour_nat)
hold on
temp = Q_CONV_out_monthly(:,:,plot_HPP);
plot(1:length(temp(:)),temp(:),'Color',colour_CONV)
temp = Q_BAL_out_monthly(:,:,plot_HPP);
plot(1:length(temp(:)),temp(:),'Color',colour_BAL)
if STOR_break(plot_HPP) == 0
    temp = Q_STOR_out_monthly(:,:,plot_HPP);
    plot([1:length(temp(:))],temp(:),'Color',colour_STOR,'LineStyle','--')
    legend '$Q_{in}$' '$Q_{out,CONV}$' '$Q_{out,BAL}$' '$Q_{out,STOR}$'
else
    legend '$Q_{in}$' '$Q_{out,CONV}$' '$Q_{out,BAL}$'
end
xticks(1:length(temp(:)))
xticklabels(months_byyear(:))
xtickangle(90)
xlim([1 length(temp(:))])
xlabel 'time (months)'
ylabel '$Q(t)$ (m$^3$/s)'
title 'inflow vs. outflow (monthly)'


% [figure] (cf. Fig. S4a, S9a)

% [find] median year in terms of ELCC
temp_ELCC = ELCC_BAL_byyear(:,plot_HPP);
y_plot_median = find(temp_ELCC == median(temp_ELCC),1);
clear temp_ELCC

% [plot] multi-year average monthly power mix in median year
figure()
subplot(1,2,1)
area_mix_BAL_bymonth = [E_hydro_BAL_stable_bymonth(:,y_plot_median(end),plot_HPP)'; E_hydro_BAL_flexible_bymonth(:,y_plot_median(end),plot_HPP)'; E_wind_BAL_bymonth(:,y_plot_median(end),plot_HPP)'; E_solar_BAL_bymonth(:,y_plot_median(end),plot_HPP)'; E_hydro_BAL_RoR_bymonth(:,y_plot_median(end),plot_HPP)']./days_year(:,y_plot_median(end))'.*10^3/hrs_day;
h = area(1:12,area_mix_BAL_bymonth','FaceColor','flat');
h(1).FaceColor = colour_hydro_stable;
h(2).FaceColor = colour_hydro_flexible;
h(3).FaceColor = colour_wind;
h(4).FaceColor = colour_solar;
h(5).FaceColor = colour_hydro_RoR;
hold on
xlim([1 months_yr])
xticks(1:months_yr)
xticklabels(months_names_full)
xtickangle(90)
ylabel 'Power generation (MWh/h)'
plot(1:months_yr,L_norm_BAL_bymonth(:,y_plot_median(end),plot_HPP).*ELCC_BAL_byyear(y_plot_median(end),plot_HPP),'k','LineWidth',3)
legend 'Hydropower (stable)' 'Hydropower (flexible)' 'Wind power' 'Solar power' 'Hydropower (RoR)' 'Load followed (ELCC)'
title('monthly generation in median year (BAL)')

% [plot] multi-year average monthly power mix in user-selected year
subplot(1,2,2)
area_mix_BAL_bymonth = [E_hydro_BAL_stable_bymonth(:,plot_year,plot_HPP)'; E_hydro_BAL_flexible_bymonth(:,plot_year,plot_HPP)'; E_wind_BAL_bymonth(:,plot_year,plot_HPP)'; E_solar_BAL_bymonth(:,plot_year,plot_HPP)'; E_hydro_BAL_RoR_bymonth(:,plot_year,plot_HPP)']./days_year(:,plot_year)'.*10^3/hrs_day;
h = area(1:12,area_mix_BAL_bymonth','FaceColor','flat');
h(1).FaceColor = colour_hydro_stable;
h(2).FaceColor = colour_hydro_flexible;
h(3).FaceColor = colour_wind;
h(4).FaceColor = colour_solar;
h(5).FaceColor = colour_hydro_RoR;
hold on
xlim([1 months_yr])
xticks(1:months_yr)
xticklabels(months_names_full)
xtickangle(90)
ylabel 'Power generation (MWh/h)'
plot(1:months_yr,L_norm_BAL_bymonth(:,plot_year,plot_HPP).*ELCC_BAL_byyear(plot_year,plot_HPP),'k','LineWidth',3)
legend  'Hydropower (stable)' 'Hydropower (flexible)' 'Wind power' 'Solar power' 'Hydropower (RoR)' 'Load followed (ELCC)'
title(strcat('monthly generation in ', '$\mbox{ }$', num2str(simulation_years(plot_year)),'$\mbox{ }$(BAL)'))


% [figure] (cf. Fig. S4b, S9b)
% [plot] power mix by year
figure()
E_generated_BAL_bymonth_sum = [sum(E_hydro_BAL_stable_bymonth(:,:,plot_HPP),1); sum(E_hydro_BAL_flexible_bymonth(:,:,plot_HPP),1); sum(E_wind_BAL_bymonth(:,:,plot_HPP),1); sum(E_solar_BAL_bymonth(:,:,plot_HPP),1); sum(E_hydro_BAL_RoR_bymonth(:,:,plot_HPP),1)];
yyaxis left
h = bar(simulation_years,E_generated_BAL_bymonth_sum','stacked');
h(1).FaceColor = colour_hydro_stable;
h(2).FaceColor = colour_hydro_flexible;
h(3).FaceColor = colour_wind;
h(4).FaceColor = colour_solar;
h(5).FaceColor = colour_hydro_RoR;
xlim([simulation_years(1) - 1 simulation_years(end) + 1])
xticks(simulation_years(1):simulation_years(end))
xticklabels(simulation_years(1):simulation_years(end))
ylim([0 max(sum(E_generated_BAL_bymonth_sum,1))*1.1])
ylabel 'Power generation (GWh/year)'
yyaxis right
plot(simulation_years,ELCC_BAL_yearly(:,plot_HPP)./10^3,'k','LineWidth',2)
ylim([0 max(sum(E_generated_BAL_bymonth_sum,1))*1.1])
legend  'Hydropower (stable)' 'Hydropower (flexible)' 'Wind power' 'Solar power' 'Hydropower (RoR)' 'Load followed (ELCC)'
title 'Multiannual generation (BAL)'

% [figure] (cf. Fig. 2 main paper, Fig. S5)
% [plot] power mix for selected days of selected month
figure()
area_mix_full = [P_BAL_hydro_stable_hourly(hrs_year,plot_year,plot_HPP)' ; P_BAL_hydro_flexible_hourly(hrs_year,plot_year,plot_HPP)'; P_BAL_wind_hourly(hrs_year,plot_year,plot_HPP)'; P_BAL_solar_hourly(hrs_year,plot_year,plot_HPP)'; P_BAL_hydro_RoR_hourly(hrs_year,plot_year,plot_HPP)'];
h = area(hrs_year - 1,area_mix_full','FaceColor','flat');
h(1).FaceColor = colour_hydro_stable;
h(2).FaceColor = colour_hydro_flexible;
h(3).FaceColor = colour_wind;
h(4).FaceColor = colour_solar;
h(5).FaceColor = colour_hydro_RoR;
hold on
ELCC_BAL_byday = P_followed_BAL_range(plot_year,P_followed_BAL_index(plot_year,plot_HPP),plot_HPP).*L_norm(hrs_year,plot_year,plot_HPP);
plot(hrs_year - 1,ELCC_BAL_byday,'k','LineWidth',2)
xlim([hrs_day*plot_day_load hrs_day*(plot_day_load + plot_num_days)])
xticks(hrs_year(1) - 1:hrs_day:hrs_year(end))
xticklabels(str_axis)
ylim([0 max(sum(area_mix_full,1)).*1.1])
legend  'Hydropower (stable)' 'Hydropower (flexible)' 'Wind power' 'Solar power' 'Hydropower (RoR)' 'Load followed (ELCC)'
xlabel 'Day of the year'
ylabel 'Power generation (MWh/h)'
title 'Daily generation \& load profiles (BAL)'

% [figure] if STOR scenario available
% STOR
if STOR_break(plot_HPP) == 0
    
    figure()
    subplot(1,2,1)
    area_mix_STOR_bymonth = [E_hydro_STOR_stable_bymonth(:,y_plot_median(end),plot_HPP)'; E_hydro_STOR_flexible_bymonth(:,y_plot_median(end),plot_HPP)'; E_wind_STOR_bymonth(:,y_plot_median(end),plot_HPP)'; E_solar_STOR_bymonth(:,y_plot_median(end),plot_HPP)'; -1.*E_hydro_pump_STOR_bymonth(:,y_plot_median(end),plot_HPP)']./days_year(:,y_plot_median(end))'.*10^3/hrs_day;
    h = area(1:12,area_mix_STOR_bymonth','FaceColor','flat');
    h(1).FaceColor = colour_hydro_stable;
    h(2).FaceColor = colour_hydro_flexible;
    h(3).FaceColor = colour_wind;
    h(4).FaceColor = colour_solar;
    h(5).FaceColor = colour_hydro_pumped;
    hold on
    xlim([1 months_yr])
    xticks(1:months_yr)
    xticklabels(months_names_full)
    xtickangle(90)
    ylabel 'Power generation (MWh/h)'
    plot(1:months_yr,L_norm_STOR_bymonth(:,y_plot_median(end),plot_HPP).*ELCC_STOR_byyear(y_plot_median(end),plot_HPP),'k','LineWidth',3)
    legend 'Hydropower (stable)' 'Hydropower (flexible)' 'Wind power' 'Solar power' 'Hydropower (RoR)' 'Load followed (ELCC)'
    title('monthly generation in median year (STOR)')
    
    % [plot] multi-year average monthly power mix in user-selected year
    subplot(1,2,2)
    area_mix_STOR_bymonth = [E_hydro_STOR_stable_bymonth(:,plot_year,plot_HPP)'; E_hydro_STOR_flexible_bymonth(:,plot_year,plot_HPP)'; E_wind_STOR_bymonth(:,plot_year,plot_HPP)'; E_solar_STOR_bymonth(:,plot_year,plot_HPP)'; -1.*E_hydro_pump_STOR_bymonth(:,plot_year,plot_HPP)']./days_year(:,plot_year)'.*10^3/hrs_day;
    h = area(1:12,area_mix_STOR_bymonth','FaceColor','flat');
    h(1).FaceColor = colour_hydro_stable;
    h(2).FaceColor = colour_hydro_flexible;
    h(3).FaceColor = colour_wind;
    h(4).FaceColor = colour_solar;
    h(5).FaceColor = colour_hydro_pumped;
    hold on
    xlim([1 months_yr])
    xticks(1:months_yr)
    xticklabels(months_names_full)
    xtickangle(90)
    ylabel 'Power generation (MWh/h)'
    plot(1:months_yr,L_norm_STOR_bymonth(:,plot_year,plot_HPP).*ELCC_STOR_byyear(plot_year,plot_HPP),'k','LineWidth',3)
    legend  'Hydropower (stable)' 'Hydropower (flexible)' 'Wind power' 'Solar power' 'Hydropower (RoR)' 'Load followed (ELCC)'
    title(strcat('monthly generation in ', '$\mbox{ }$', num2str(simulation_years(plot_year)),'$\mbox{ }$(STOR)'))
    
    
    % [figure] (cf. Fig. S4b, S9b)
    % [plot] power mix by year
    figure()
    E_generated_STOR_bymonth_sum = [sum(E_hydro_STOR_stable_bymonth(:,:,plot_HPP),1); sum(E_hydro_STOR_flexible_bymonth(:,:,plot_HPP),1); sum(E_wind_STOR_bymonth(:,:,plot_HPP),1); sum(E_solar_STOR_bymonth(:,:,plot_HPP),1); -1.*sum(E_hydro_pump_STOR_bymonth(:,:,plot_HPP),1)];
    yyaxis left
    h = bar(simulation_years,E_generated_STOR_bymonth_sum','stacked');
    h(1).FaceColor = colour_hydro_stable;
    h(2).FaceColor = colour_hydro_flexible;
    h(3).FaceColor = colour_wind;
    h(4).FaceColor = colour_solar;
    h(5).FaceColor = colour_hydro_pumped;
    xlim([simulation_years(1) - 1 simulation_years(end) + 1])
    xticks(simulation_years(1):simulation_years(end))
    xticklabels(simulation_years(1):simulation_years(end))
    ylim([0 max(sum(E_generated_STOR_bymonth_sum,1))*1.1])
    ylabel 'Power generation (GWh/year)'
    yyaxis right
    plot(simulation_years,ELCC_STOR_yearly(:,plot_HPP)./10^3,'k','LineWidth',2)
    ylim([0 max(sum(E_generated_STOR_bymonth_sum,1))*1.1])
    legend  'Hydropower (stable)' 'Hydropower (flexible)' 'Wind power' 'Solar power' 'Excess RE stored by pumping' 'Load followed (ELCC)'
    title 'Multiannual generation (STOR)'
    
    figure()
    area_mix_full = [P_STOR_hydro_stable_hourly(hrs_year,plot_year,plot_HPP)'; P_STOR_hydro_flexible_hourly(hrs_year,plot_year,plot_HPP)'; P_STOR_wind_hourly(hrs_year,plot_year,plot_HPP)'; P_STOR_solar_hourly(hrs_year,plot_year,plot_HPP)'; -1.*P_STOR_pump_hourly(hrs_year,plot_year,plot_HPP)';];
    h = area(hrs_year - 1,area_mix_full','FaceColor','flat');
    h(1).FaceColor = colour_hydro_stable;
    h(2).FaceColor = colour_hydro_flexible;
    h(3).FaceColor = colour_wind;
    h(4).FaceColor = colour_solar;
    h(5).FaceColor = colour_hydro_pumped;
    hold on
    ELCC_STOR_byday = P_followed_STOR_range(plot_year,P_followed_STOR_index(plot_year,plot_HPP),plot_HPP).*L_norm(hrs_year,plot_year,plot_HPP);
    plot(hrs_year - 1,ELCC_STOR_byday,'k','LineWidth',2)
    xlim([hrs_day*plot_day_load hrs_day*(plot_day_load + plot_num_days)])
    xticks(hrs_year(1) - 1:hrs_day:hrs_year(end))
    xticklabels(str_axis)
    legend  'Hydropower (stable)' 'Hydropower (flexible)' 'Wind power' 'Solar power' 'Excess RE stored by pumping' 'Load followed (ELCC)'
    xlabel 'day of the year'
    ylabel 'Power generation (MWh/h)'
    title 'Daily generation \& load profiles (STOR)'
end


% [preallocate] vectors for adapted rules under BAL
h_BAL_rules_stable_bymonth = zeros(months_yr,length(simulation_years));
h_BAL_rules_total_bymonth = zeros(months_yr,length(simulation_years),max(max(days_year)),hrs_day);
Q_out_net_BAL_rules_stable_bymonth = NaN.*ones(months_yr,length(simulation_years));
Q_out_net_BAL_rules_total_bymonth = NaN.*ones(months_yr,length(simulation_years),max(max(days_year)),hrs_day);

% [loop] across all simulation years to calculate release for each day, hour, month and year
for y = 1:length(simulation_years)
    
    % [loop] across all months in each year
    for m = 1:months_yr
        
        % [arrange] hourly head and outflow values for each month
        temp_head_BAL_bymonth = h_BAL_hourly(positions(m,y):positions(m+1,y) - 1,y,plot_HPP);
        temp_Q_BAL_bymonth = Q_BAL_out_hourly(positions(m,y):positions(m+1,y) - 1,y,plot_HPP) - Q_in_RoR_hourly(positions(m,y):positions(m+1,y) - 1,y,plot_HPP);
        
        % [arrange] get unique values (because head determination happens at accuracy given by N_calibration)
        temp_head_unique = unique(temp_head_BAL_bymonth);
        temp_Q_unique = unique(temp_Q_BAL_bymonth);
        
        % [preallocate] minimum environmental flow rule
        temp_Q_out_net_BAL_rules_stable = zeros(1,length(temp_head_unique));
        
        % [loop] to find minimum flow
        for n = 1:length(temp_head_unique)
            temp_head = temp_head_BAL_bymonth(temp_head_BAL_bymonth == temp_head_unique(n));
            temp_Q_out_rules = temp_Q_BAL_bymonth(temp_head_BAL_bymonth == temp_head_unique(n));
            temp_Q_out_net_BAL_rules_stable(n) = min(temp_Q_out_rules);
        end
        
        % [calculate] average head and outflow in each month
        h_BAL_rules_stable_bymonth(m,y) = mean(temp_head_BAL_bymonth);
        Q_out_net_BAL_rules_stable_bymonth(m,y) = min(temp_Q_out_net_BAL_rules_stable);
        
        % [arrange] according to specific hours of day
        % [loop] across all hours of the day
        for hr = 1:hrs_day
            
            % [find] head during specific hours
            temp_head_bymonth_hr = temp_head_BAL_bymonth(hr:hrs_day:end);
            
            % [find] outflow during specific hours
            temp_Q_bymonth_hr = temp_Q_BAL_bymonth(hr:hrs_day:end);
            
            % [loop] across all days of the month to find rules for flexible outflow
            for day = 1:days_year(m,y)
                
                % [calculate] head and outflow for each hour on each day
                temp_h_BAL_rules_flexible = temp_head_bymonth_hr(day);
                temp_Q_out_net_BAL_rules_flexible = temp_Q_bymonth_hr(day);
                
                % [arrange] head and outflow for each hour, day, month, year
                h_BAL_rules_total_bymonth(m,y,day,hr) = temp_h_BAL_rules_flexible;
                Q_out_net_BAL_rules_total_bymonth(m,y,day,hr) = temp_Q_out_net_BAL_rules_flexible;
                
            end
        end
    end
end


% [figure] approximate release rules for selected months and hours of day (Fig. 2b main paper)
figure()

% [initialise] legend index
clear legendItem
legendIndex = 0;


% [loop] across selected hours of day
for hr = plot_rules_hr
    
    % [loop] across selected months
    for m = plot_rules_month
        
        
        % [calculate] new legend index
        legendIndex = legendIndex + 1;
        
        % [preallocate] temporary vectors to store head and outflow
        temp_h = zeros(1,length(simulation_years));
        temp_Q = zeros(1,length(simulation_years));
        temp_Q_std = zeros(1,length(simulation_years));
        
        % [loop] across all simulation years
        for y = 1:length(simulation_years)
            
            % [calculate] head at given hour of day for all days in a single month in a single year
            temp = h_BAL_rules_total_bymonth(m,y,1:days_year(m,y),hr);
            % [calculate] take the mean for that time of day in that month
            temp_h(y) = nanmean(temp(:));
            
            % [calculate] outflow at given hour of day for all days in a single month in a single year
            temp = Q_out_net_BAL_rules_total_bymonth(m,y,1:days_year(m,y),hr);
            % [calculate] take the mean for that time of day in that month
            temp_Q(y) = nanmean(temp(:));
            temp_Q_std(y) = nanstd(temp(:));
            
            % [check] mark drought incidences
            if hydro_BAL_curtailment_factor_monthly(m,y,plot_HPP) == 0
                temp_Q(y) = NaN;
                temp_h(y) = NaN;
                temp_Q_std(y) = NaN;
            end
        end
        
        % [check] remove drought incidences
        temp_Q(isnan(temp_Q)) = [];
        temp_h(isnan(temp_h)) = [];
        temp_Q_std(isnan(temp_Q_std)) = [];
        
        errorbar(temp_h,temp_Q,temp_Q_std,'^','LineWidth',2)
        legendItem(legendIndex) = strcat('total outflow$\mbox{ }$', [num2str(hr) 'h$\mbox{ }$'], months_names_full(m));
        legendIndex = legendIndex + 1;
        hold on
        [temp_fit, ~] = polyfit(temp_h,temp_Q,1);
        legendItem(legendIndex) = strcat('total outflow$\mbox{ }$', [num2str(hr) 'h$\mbox{ }$'], months_names_full(m), '$\mbox{ }$fit');
        plot(temp_h,temp_fit(2) + temp_fit(1).*temp_h,'k--')
        
    end
end
plot([nanmin(nanmin(h_BAL_hourly(:,:,plot_HPP))) nanmax(nanmax(h_BAL_hourly(:,:,plot_HPP)))], ones(1,2).*nanmean(nanmean(Q_BAL_stable_hourly(:,:,plot_HPP))),'k')
legendItem(legendIndex + 1) = 'stable outflow';
legend(legendItem)
xlabel 'Hydraulic head (m)'
ylabel '$Q_{out}$ (m$^3$/s)'
title('release rules (BAL)')
