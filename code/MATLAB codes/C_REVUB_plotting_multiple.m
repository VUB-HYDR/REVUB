%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% REVUB plotting results %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% © 2019 CIREG project
% Author: Sebastian Sterl, Vrije Universiteit Brussel
% This code accompanies the paper "Smart renewable electricity portfolios in West Africa" by Sterl et al.
% All equation, section &c. numbers refer to that paper and its Supplementary Materials, unless otherwise mentioned.

set(groot, 'defaultTextInterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
set(0,'DefaultAxesFontSize',8)
set(0,'DefaultLineLineWidth',1)

% [initialize] close windows from previous plotting
close all

% [set by user] select hydropower plant and year, month, days for which to display results
plot_HPP_multiple = [1 2];
plot_year_multiple = 15;
plot_month_multiple = 4;
plot_day_month_multiple = 2;
plot_num_days_multiple = 3;

% [set by user] total electricity demand to be met (MW) - these numbers are chosen for illustrative purposes only
P_total_av = 400;
P_total_hourly = P_total_av.*L_norm(:,:,1);

% [initialise] use STOR equal to BAL for reservoirs where STOR not modelled
for HPP = plot_HPP_multiple
    if STOR_break(HPP) == 1
        P_STOR_hydro_stable_hourly(:,:,HPP) = P_BAL_hydro_stable_hourly(:,:,HPP);
        P_STOR_hydro_flexible_hourly(:,:,HPP) = P_BAL_hydro_flexible_hourly(:,:,HPP);
        P_STOR_wind_hourly(:,:,HPP) = P_BAL_wind_hourly(:,:,HPP);
        P_STOR_solar_hourly(:,:,HPP) = P_BAL_solar_hourly(:,:,HPP); 
        P_STOR_pump_hourly(:,:,HPP) = 0;
        ELCC_STOR_yearly(:,HPP) = ELCC_BAL_yearly(:,HPP);
        L_followed_STOR_hourly(:,:,HPP) = L_followed_BAL_hourly(:,:,HPP);
    end
end

% [calculate] non-hydro-solar-wind (thermal) power contribution (difference between total and hydro-solar-wind)
P_BAL_thermal_hourly = P_total_hourly - nansum(P_BAL_hydro_stable_hourly(:,:,plot_HPP_multiple) + P_BAL_hydro_flexible_hourly(:,:,plot_HPP_multiple) + P_BAL_wind_hourly(:,:,plot_HPP_multiple) + P_BAL_solar_hourly(:,:,plot_HPP_multiple) + P_BAL_hydro_RoR_hourly(:,:,plot_HPP_multiple),3);
P_STOR_thermal_hourly = P_total_hourly - nansum(P_STOR_hydro_stable_hourly(:,:,plot_HPP_multiple) + P_STOR_hydro_flexible_hourly(:,:,plot_HPP_multiple) + P_STOR_wind_hourly(:,:,plot_HPP_multiple) + P_STOR_solar_hourly(:,:,plot_HPP_multiple) + P_BAL_hydro_RoR_hourly(:,:,plot_HPP_multiple) - P_STOR_pump_hourly(:,:,plot_HPP_multiple),3);
P_BAL_thermal_hourly(P_BAL_thermal_hourly < 0) = 0;
P_STOR_thermal_hourly(P_STOR_thermal_hourly < 0) = 0;

% [calculate] excess (to-be-curtailed) power
P_BAL_curtailed_hourly = nansum(P_BAL_hydro_stable_hourly(:,:,plot_HPP_multiple) + P_BAL_hydro_flexible_hourly(:,:,plot_HPP_multiple) + P_BAL_wind_hourly(:,:,plot_HPP_multiple) + P_BAL_solar_hourly(:,:,plot_HPP_multiple) + P_BAL_hydro_RoR_hourly(:,:,plot_HPP_multiple),3) + P_BAL_thermal_hourly - P_total_hourly;
P_STOR_curtailed_hourly = nansum(P_STOR_hydro_stable_hourly(:,:,plot_HPP_multiple) + P_STOR_hydro_flexible_hourly(:,:,plot_HPP_multiple) + P_STOR_wind_hourly(:,:,plot_HPP_multiple) + P_STOR_solar_hourly(:,:,plot_HPP_multiple) + P_BAL_hydro_RoR_hourly(:,:,plot_HPP_multiple) - P_STOR_pump_hourly(:,:,plot_HPP_multiple),3) + P_STOR_thermal_hourly - P_total_hourly;

% [read] vector with hours in each year
hrs_year = 1:hrs_byyear(plot_year_multiple);

% [identify] index of day of month to plot
plot_day_load = sum(days_year(1:plot_month_multiple - 1,plot_year_multiple)) + plot_day_month_multiple - 1;

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
            str(d,m,y) = strcat(num2str(d), months_names_full(m), 'Yr', num2str(y));
        end
    end
end
str_axis = str(:,:,plot_year_multiple);
str_axis(ismissing(str_axis)) = [];

% [colours] for plotting
colour_hydro_stable = [55, 126, 184] / 255;
colour_hydro_flexible = [106, 226, 207] / 255;
colour_solar = [255, 255, 51] / 255;
colour_wind = [77, 175, 74] / 255;
colour_hydro_RoR = [100, 100, 100] / 255;
colour_hydro_pumped = [77, 191, 237] / 255;
colour_thermal = [75, 75, 75] / 255;
colour_curtailed = [200, 200, 200] / 255;

% [preallocate] to aggregate output variables by month for CONV
E_CONV_stable_bymonth = zeros(months_yr,length(simulation_years),HPP_number);

% [preallocate] to aggregate output variables by month for BAL
L_norm_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
E_hydro_BAL_stable_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
E_solar_BAL_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
E_wind_BAL_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
E_hydro_BAL_flexible_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
E_hydro_BAL_RoR_bymonth = zeros(months_yr,length(simulation_years),HPP_number);

% [preallocate] to aggregate output variables by month for STOR
E_hydro_STOR_stable_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
E_solar_STOR_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
E_wind_STOR_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
E_hydro_STOR_flexible_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
E_hydro_pump_STOR_bymonth = zeros(months_yr,length(simulation_years),HPP_number);

% [preallocate] to plot ELCC at monthly timestep
ELCC_BAL_bymonth = zeros(months_yr,length(simulation_years),HPP_number);
ELCC_STOR_bymonth = zeros(months_yr,length(simulation_years),HPP_number);

% [preallocate] extra variables for thermal power generation assessment
E_total_bymonth = zeros(months_yr,length(simulation_years));
E_thermal_BAL_bymonth = zeros(months_yr,length(simulation_years));
E_thermal_STOR_bymonth = zeros(months_yr,length(simulation_years));
E_curtailed_BAL_bymonth = zeros(months_yr,length(simulation_years));
E_curtailed_STOR_bymonth = zeros(months_yr,length(simulation_years));


% [loop] across all hydropower plants to aggregate output variables by month
for HPP = 1:HPP_number
    
    % [loop] across all years in the simulation
    for y = 1:length(simulation_years)
        
        % [loop] across all months of the year
        for m = 1:months_yr
            
            % [calculate] power generation, converting hourly values (MW or MWh/h) to GWh/month
            E_CONV_stable_bymonth(m,y,HPP) = 1e-3.*sum(P_CONV_hydro_stable_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            
            E_hydro_BAL_stable_bymonth(m,y,HPP) = 1e-3.*sum(P_BAL_hydro_stable_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            E_solar_BAL_bymonth(m,y,HPP) = 1e-3.*sum(P_BAL_solar_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            E_wind_BAL_bymonth(m,y,HPP) = 1e-3.*sum(P_BAL_wind_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            E_hydro_BAL_flexible_bymonth(m,y,HPP) = 1e-3.*sum(P_BAL_hydro_flexible_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            E_hydro_BAL_RoR_bymonth(m,y,HPP) = 1e-3.*sum(P_BAL_hydro_RoR_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            L_norm_bymonth(m,y,HPP) = mean(L_norm(sum(days_year(1:m-1,y))*hrs_day + 1 : sum(days_year(1:m,y))*hrs_day,y,HPP));

            E_hydro_STOR_stable_bymonth(m,y,HPP) = 1e-3.*sum(P_STOR_hydro_stable_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            E_solar_STOR_bymonth(m,y,HPP) = 1e-3.*sum(P_STOR_solar_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            E_wind_STOR_bymonth(m,y,HPP) = 1e-3.*sum(P_STOR_wind_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            E_hydro_STOR_flexible_bymonth(m,y,HPP) = 1e-3.*sum(P_STOR_hydro_flexible_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            E_hydro_pump_STOR_bymonth(m,y,HPP) = 1e-3.*sum(P_STOR_pump_hourly(positions(m,y):positions(m+1,y)-1,y,HPP));
            
            % [calculate] ELCC by month (MWh/h)
            ELCC_BAL_bymonth(m,y,HPP) = sum(L_followed_BAL_hourly(positions(m,y):positions(m+1,y)-1,y,HPP))./days_year(m,y)'/hrs_day;
            ELCC_STOR_bymonth(m,y,HPP) = sum(L_followed_STOR_hourly(positions(m,y):positions(m+1,y)-1,y,HPP))./days_year(m,y)'/hrs_day;
            
        end
         
    end

end

% [loop] across all years in the simulation
for y = 1:length(simulation_years) 
    % [loop] across all months of the year, converting hourly values (MW or MWh/h) to GWh/month (see eq. S24, S25)
    for m = 1:months_yr
        E_total_bymonth(m,y) = 1e-3.*sum(P_total_hourly(positions(m,y):positions(m+1,y)-1,y));
        E_thermal_BAL_bymonth(m,y) = 1e-3.*sum(P_BAL_thermal_hourly(positions(m,y):positions(m+1,y)-1,y));
        E_thermal_STOR_bymonth(m,y) = 1e-3.*sum(P_STOR_thermal_hourly(positions(m,y):positions(m+1,y)-1,y));   
        E_curtailed_BAL_bymonth(m,y) = 1e-3.*sum(P_BAL_curtailed_hourly(positions(m,y):positions(m+1,y)-1,y));
        E_curtailed_STOR_bymonth(m,y) = 1e-3.*sum(P_STOR_curtailed_hourly(positions(m,y):positions(m+1,y)-1,y));     
    end
end


% [figure] (cf. Fig. S4a, S9a)
% [plot] average monthly power mix in user-selected year
figure()
area_mix_BAL_bymonth = [nansum(E_hydro_BAL_stable_bymonth(:,plot_year_multiple,plot_HPP_multiple),3)'; nansum(E_hydro_BAL_flexible_bymonth(:,plot_year_multiple,plot_HPP_multiple),3)'; nansum(E_wind_BAL_bymonth(:,plot_year_multiple,plot_HPP_multiple),3)'; nansum(E_solar_BAL_bymonth(:,plot_year_multiple,plot_HPP_multiple),3)' - E_curtailed_BAL_bymonth(:,plot_year_multiple)'; nansum(E_hydro_BAL_RoR_bymonth(:,plot_year_multiple,plot_HPP_multiple),3)'; E_thermal_BAL_bymonth(:,plot_year_multiple)'; E_curtailed_BAL_bymonth(:,plot_year_multiple)']./days_year(:,plot_year_multiple)'.*10^3/hrs_day;
h = area(1:12,area_mix_BAL_bymonth','FaceColor','flat');
h(1).FaceColor = colour_hydro_stable;
h(2).FaceColor = colour_hydro_flexible;
h(3).FaceColor = colour_wind;
h(4).FaceColor = colour_solar;
h(5).FaceColor = colour_hydro_RoR;
h(6).FaceColor = colour_thermal;
h(7).FaceColor = colour_curtailed;
hold on
xlim([1 months_yr])
xticks(1:months_yr)
xticklabels(months_names_full)
xtickangle(90)
ylabel 'Power generation (MWh/h)'
plot(1:months_yr,E_total_bymonth(:,plot_year_multiple)./days_year(:,plot_year_multiple).*10^3/hrs_day,'k','LineWidth',3)
plot(1:months_yr,nansum(ELCC_BAL_bymonth(:,plot_year_multiple,plot_HPP_multiple),3),'k--','LineWidth',3)
legend  'Hydropower (stable)' 'Hydropower (flexible)' 'Wind power' 'Solar power' 'Hydropower (RoR)' 'Thermal' 'Curtailed VRE' 'Total load' 'ELCC$_{tot}$'
title(strcat('monthly generation in ', '$\mbox{ }$', 'year', '$\mbox{ }$', num2str(plot_year_multiple),'$\mbox{ }$(BAL)'))


% [figure] (cf. Fig. S4b, S9b)
% [plot] power mix by year
figure()
E_generated_BAL_bymonth_sum = [nansum(sum(E_hydro_BAL_stable_bymonth(:,:,plot_HPP_multiple),1),3); nansum(sum(E_hydro_BAL_flexible_bymonth(:,:,plot_HPP_multiple),1),3); nansum(sum(E_wind_BAL_bymonth(:,:,plot_HPP_multiple),1),3); nansum(sum(E_solar_BAL_bymonth(:,:,plot_HPP_multiple),1),3) - sum(E_curtailed_BAL_bymonth,1); nansum(sum(E_hydro_BAL_RoR_bymonth(:,:,plot_HPP_multiple),1),3); sum(E_thermal_BAL_bymonth,1); sum(E_curtailed_BAL_bymonth,1)];
yyaxis left
h = bar(simulation_years,E_generated_BAL_bymonth_sum','stacked');
h(1).FaceColor = colour_hydro_stable;
h(2).FaceColor = colour_hydro_flexible;
h(3).FaceColor = colour_wind;
h(4).FaceColor = colour_solar;
h(5).FaceColor = colour_hydro_RoR;
h(6).FaceColor = colour_thermal;
h(7).FaceColor = colour_curtailed;
xlim([simulation_years(1) - 1 simulation_years(end) + 1])
xticks(simulation_years(1):simulation_years(end))
xticklabels([1:length(simulation_years)])
xlabel 'year'
ylim([0 max(sum(E_generated_BAL_bymonth_sum,1))*1.1])
ylabel 'Power generation (GWh/year)'
yyaxis right
hold on
plot(simulation_years,sum(E_total_bymonth,1),'k','LineWidth',2)
plot(simulation_years,sum(ELCC_BAL_yearly(:,plot_HPP_multiple),2)./10^3,'k--','LineWidth',2)
ylim([0 max(sum(E_generated_BAL_bymonth_sum,1))*1.1])
legend  'Hydropower (stable)' 'Hydropower (flexible)' 'Wind power' 'Solar power' 'Hydropower (RoR)' 'Thermal' 'Curtailed VRE' 'Total load' 'ELCC$_{tot}$'
title 'Multiannual generation (BAL)'

% [figure] (cf. Fig. 2 main paper, Fig. S5)
% [plot] power mix for selected days of selected month
figure()
area_mix_full = [nansum(P_BAL_hydro_stable_hourly(hrs_year,plot_year_multiple,plot_HPP_multiple),3)' ; nansum(P_BAL_hydro_flexible_hourly(hrs_year,plot_year_multiple,plot_HPP_multiple),3)'; nansum(P_BAL_wind_hourly(hrs_year,plot_year_multiple,plot_HPP_multiple),3)'; nansum(P_BAL_solar_hourly(hrs_year,plot_year_multiple,plot_HPP_multiple),3)'; nansum(P_BAL_hydro_RoR_hourly(hrs_year,plot_year_multiple,plot_HPP_multiple),3)'; P_BAL_thermal_hourly(hrs_year,plot_year_multiple)'; -1.*P_BAL_curtailed_hourly(hrs_year,plot_year_multiple)'];
h = area(hrs_year - 1,area_mix_full','FaceColor','flat');
h(1).FaceColor = colour_hydro_stable;
h(2).FaceColor = colour_hydro_flexible;
h(3).FaceColor = colour_wind;
h(4).FaceColor = colour_solar;
h(5).FaceColor = colour_hydro_RoR;
h(6).FaceColor = colour_thermal;
h(7).FaceColor = colour_curtailed;
hold on
plot(hrs_year - 1, P_total_hourly(hrs_year,plot_year_multiple),'k','LineWidth',2)
plot(hrs_year - 1, nansum(L_followed_BAL_hourly(hrs_year,plot_year_multiple,plot_HPP_multiple),3),'k--','LineWidth',2)
xlim([hrs_day*plot_day_load hrs_day*(plot_day_load + plot_num_days_multiple)])
xticks(hrs_year(1) - 1:hrs_day:hrs_year(end))
xticklabels(str_axis)
ylim([0 max(sum(area_mix_full,1)).*1.1])
legend  'Hydropower (stable)' 'Hydropower (flexible)' 'Wind power' 'Solar power' 'Hydropower (RoR)' 'Thermal' 'Curtailed VRE' 'Total load' 'ELCC$_{tot}$'
xlabel 'Day of the year'
ylabel 'Power generation (MWh/h)'
title 'Daily generation \& load profiles (BAL)'

% [check] if STOR scenario available
if option_storage == 1 && min(STOR_break(plot_HPP_multiple)) == 0
    
    % [figure] (cf. Fig. S4a, S9a)
    % [plot] average monthly power mix in user-selected year
    figure()
    area_mix_STOR_bymonth = [nansum(E_hydro_STOR_stable_bymonth(:,plot_year_multiple,plot_HPP_multiple),3)'; nansum(E_hydro_STOR_flexible_bymonth(:,plot_year_multiple,plot_HPP_multiple),3)'; nansum(E_wind_STOR_bymonth(:,plot_year_multiple,plot_HPP_multiple),3)'; nansum(E_solar_STOR_bymonth(:,plot_year_multiple,plot_HPP_multiple) - E_hydro_pump_STOR_bymonth(:,plot_year_multiple,plot_HPP_multiple),3)' - E_curtailed_STOR_bymonth(:,plot_year_multiple)'; nansum(E_hydro_BAL_RoR_bymonth(:,plot_year_multiple,plot_HPP_multiple),3)'; E_thermal_STOR_bymonth(:,plot_year_multiple)'; E_curtailed_STOR_bymonth(:,plot_year_multiple)';]./days_year(:,plot_year_multiple)'.*10^3/hrs_day;
    h = area(1:12,area_mix_STOR_bymonth','FaceColor','flat');
    hold on
    h_neg = area(1:12,-1.*nansum(E_hydro_pump_STOR_bymonth(:,plot_year_multiple,plot_HPP_multiple),3),'FaceColor','flat');
    h(1).FaceColor = colour_hydro_stable;
    h(2).FaceColor = colour_hydro_flexible;
    h(3).FaceColor = colour_wind;
    h(4).FaceColor = colour_solar;
    h(5).FaceColor = colour_hydro_RoR;
    h(6).FaceColor = colour_thermal;
    h(7).FaceColor = colour_curtailed;
    h_neg(1).FaceColor = colour_hydro_pumped;
    hold on
    xlim([1 months_yr])
    xticks(1:months_yr)
    xticklabels(months_names_full)
    xtickangle(90)
    ylabel 'Power generation (MWh/h)'
    plot(1:months_yr,E_total_bymonth(:,plot_year_multiple)./days_year(:,plot_year_multiple).*10^3/hrs_day,'k','LineWidth',3)
    plot(1:months_yr,nansum(ELCC_STOR_bymonth(:,plot_year_multiple,plot_HPP_multiple),3),'k--','LineWidth',3)
    legend  'Hydropower (stable)' 'Hydropower (flexible)' 'Wind power' 'Solar power' 'Hydropower (RoR)' 'Thermal' 'Curtailed VRE' 'Stored VRE' 'Total load' 'ELCC$_{tot}$'
    title(strcat('monthly generation in ', '$\mbox{ }$', 'year', '$\mbox{ }$', num2str(plot_year_multiple),'$\mbox{ }$(STOR)'))
    
    
    % [figure] (cf. Fig. S4b, S9b)
    % [plot] power mix by year
    figure()
    E_generated_STOR_bymonth_sum = [nansum(sum(E_hydro_STOR_stable_bymonth(:,:,plot_HPP_multiple),1),3); nansum(sum(E_hydro_STOR_flexible_bymonth(:,:,plot_HPP_multiple),1),3); nansum(sum(E_wind_STOR_bymonth(:,:,plot_HPP_multiple),1),3); nansum(sum(E_solar_STOR_bymonth(:,:,plot_HPP_multiple) - E_hydro_pump_STOR_bymonth(:,:,plot_HPP_multiple),1),3) - sum(E_curtailed_STOR_bymonth,1); nansum(sum(E_hydro_BAL_RoR_bymonth(:,:,plot_HPP_multiple),1),3); sum(E_thermal_STOR_bymonth,1); sum(E_curtailed_STOR_bymonth,1)];
    yyaxis left
    h = bar(simulation_years,E_generated_STOR_bymonth_sum','stacked');
    hold on
    h_neg = bar(simulation_years,-1.*nansum(sum(E_hydro_pump_STOR_bymonth(:,:,plot_HPP_multiple),1),3));
    h(1).FaceColor = colour_hydro_stable;
    h(2).FaceColor = colour_hydro_flexible;
    h(3).FaceColor = colour_wind;
    h(4).FaceColor = colour_solar;
    h(5).FaceColor = colour_hydro_RoR;
    h(6).FaceColor = colour_thermal;
    h(7).FaceColor = colour_curtailed;
    h_neg(1).FaceColor = colour_hydro_pumped;
    xlim([simulation_years(1) - 1 simulation_years(end) + 1])
    xticks(simulation_years(1):simulation_years(end))
    xticklabels(1:length(simulation_years))
    ylim([min(-1.*nansum(sum(E_hydro_pump_STOR_bymonth(:,:,plot_HPP_multiple),1),3))*1.1 nanmax(sum(E_generated_STOR_bymonth_sum,1))*1.1])
    xlabel 'year'
    ylabel 'Power generation (GWh/year)'
    yyaxis right
    hold on
    plot(simulation_years,sum(E_total_bymonth,1),'k','LineWidth',2)
    plot(simulation_years,sum(ELCC_STOR_yearly(:,plot_HPP_multiple),2)./10^3,'k--','LineWidth',2)
    ylim([min(-1.*nansum(sum(E_hydro_pump_STOR_bymonth(:,:,plot_HPP_multiple),1),3))*1.1 nanmax(sum(E_generated_STOR_bymonth_sum,1))*1.1])
    legend  'Hydropower (stable)' 'Hydropower (flexible)' 'Wind power' 'Solar power' 'Hydropower (RoR)' 'Thermal' 'Curtailed VRE' 'Stored VRE' 'Total load' 'ELCC$_{tot}$'
    title 'Multiannual generation (STOR)'
    
    % [figure] (cf. Fig. 2 main paper, Fig. S5)
    % [plot] power mix for selected days of selected month
    figure()
    area_mix_full = [nansum(P_STOR_hydro_stable_hourly(hrs_year,plot_year_multiple,plot_HPP_multiple),3)'; nansum(P_STOR_hydro_flexible_hourly(hrs_year,plot_year_multiple,plot_HPP_multiple),3)'; nansum(P_STOR_wind_hourly(hrs_year,plot_year_multiple,plot_HPP_multiple),3)'; nansum(P_STOR_solar_hourly(hrs_year,plot_year_multiple,plot_HPP_multiple) - P_STOR_pump_hourly(hrs_year,plot_year_multiple,plot_HPP_multiple),3)'; nansum(P_BAL_hydro_RoR_hourly(hrs_year,plot_year_multiple,plot_HPP_multiple),3)'; P_STOR_thermal_hourly(hrs_year,plot_year_multiple)'; -1.*P_STOR_curtailed_hourly(hrs_year,plot_year_multiple)'];
    h = area(hrs_year - 1,area_mix_full','FaceColor','flat');
    hold on
    h_neg = area(hrs_year - 1,-1.*nansum(P_STOR_pump_hourly(hrs_year,plot_year_multiple,plot_HPP_multiple),3)','FaceColor','flat');
    h(1).FaceColor = colour_hydro_stable;
    h(2).FaceColor = colour_hydro_flexible;
    h(3).FaceColor = colour_wind;
    h(4).FaceColor = colour_solar;
    h(5).FaceColor = colour_hydro_RoR;
    h(6).FaceColor = colour_thermal;
    h(7).FaceColor = colour_curtailed;
    h_neg(1).FaceColor = colour_hydro_pumped;
    hold on
    plot(hrs_year - 1, P_total_hourly(hrs_year,plot_year_multiple),'k','LineWidth',2)
    plot(hrs_year - 1, nansum(L_followed_STOR_hourly(hrs_year,plot_year_multiple,plot_HPP_multiple),3),'k--','LineWidth',2)
    xlim([hrs_day*plot_day_load hrs_day*(plot_day_load + plot_num_days_multiple)])
    xticks(hrs_year(1) - 1:hrs_day:hrs_year(end))
    xticklabels(str_axis)
    legend  'Hydropower (stable)' 'Hydropower (flexible)' 'Wind power' 'Solar power' 'Hydropower (RoR)' 'Thermal' 'Curtailed VRE' 'Stored VRE' 'Total load' 'ELCC$_{tot}$'
    xlabel 'day of the year'
    ylabel 'Power generation (MWh/h)'
    title 'Daily generation \& load profiles (STOR)'
end
