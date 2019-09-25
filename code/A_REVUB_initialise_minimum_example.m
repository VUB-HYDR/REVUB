%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% REVUB initialise minimum working example %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% � 2019 CIREG project
% Author: Sebastian Sterl, Vrije Universiteit Brussel
% This code accompanies the paper "Streamlining hydro, solar and wind energy strategies in West Africa from power plant to power pool" by Sterl et al.
% All equation, section &c. numbers refer to that paper and its Supplementary Materials, unless otherwise mentioned.

% This code initialises all data needed for a minimum working example (Bui hydropower plant in Ghana and Buyo hydropower plant in C�te d'Ivoire under the REFERENCE scenario).

clc; clear all; close all


%% pre.1) Time-related parameters

% [define] number of hydropower plants in this simulation
HPP_number = 2;

% [define] The reference years used in the simulation
simulation_years = 1998:2014;

% [define] number of hours in a day
hrs_day = 24;

% [define] number of months in a year
months_yr = 12;

% [define] number of seconds and minutes in an hour
secs_hr = 3600;
mins_hr = 60;

% [preallocate] number of days in each year
days_year = zeros(months_yr,length(simulation_years));
hrs_byyear = zeros(1,length(simulation_years));

% [calculate] For each year in the simulation: determine if leap year or not;
% write corresponding amount of hours into hrs_byyear
for y = 1:length(simulation_years)
    if ceil(simulation_years(y)/4) == simulation_years(y)/4 && ceil(simulation_years(y)/100) ~= simulation_years(y)/4
        days_year(:,y) = [31 29 31 30 31 30 31 31 30 31 30 31];
    else
        days_year(:,y) = [31 28 31 30 31 30 31 31 30 31 30 31];
    end
    hrs_byyear(y) = sum(days_year(:,y))*hrs_day;
end

% [arrange] for data arrangements in matrices: determine hours corresponding to start of each month
% (e.g. January = 1; February = 745; March = 1417 or 1441 depending on whether leap year or not; &c.)
positions = zeros(size(days_year,1)+1,length(simulation_years));
positions(1,:) = 1;
for y = 1:length(simulation_years)
    for n = 1:size(days_year,1)
        positions(n+1,y) = hrs_day*days_year(n,y) + positions(n,y);
    end
end


%% pre.2) Model parameters

%%%%% GENERAL HYDROPOWER DATA %%%%%

% [set by user] wish to model storage (section S7) or not? (0 = no, 1 = yes)
option_storage = 1;

% [constant] Density of water (kg/m^3) (introduced in eq. S3)
rho = 1000;

% [constant] Gravitational acceleration (m/s^2) (introduced in eq. S8)
g = 9.81;


%%%%% HYDROPOWER OPERATION PARAMETERS %%%%%

% [set by user] Turbine efficiency (introduced in eq. S8)
eta_turb = 0.8;

% [set by user] Pumping efficiency
eta_pump = eta_turb;

% [set by user] minimum required environmental outflow fraction (eq. S4, S5)
d_min = 0.4;

% [set by user] alpha (eq. S6) for conventional HPP operation rule curve (eq. S4)
alpha = 2/3;

% [set by user] gamma (eq. S4) for conventional HPP operation rule curve (eq. S4):
gamma_hydro = 10;

% [set by user] f_opt (eq. S4, S5)
f_opt = 0.8;

% [set by user] f_spill (eq. S7)
f_spill = 0.95;

% [set by user] mu (eq. S7)
mu = 0.1;

% [set by user] Thresholds f_stop and f_restart (see page 4) for stopping and restarting
% hydropower production to maintain minimum drawdown levels
f_stop = 0.10;
f_restart = 0.20;

% [set by user] Ramp rate restrictions (eq. S16, S37): fraction of full capacity per minute
dP_ramp_turb = 12.8/5/100;
dP_ramp_pump = dP_ramp_turb;

% [set by user] Array of C_{OR} values (eq. S14). The first value is the default. If the
% criterium on k_turb (eq. S28) is not met, the simulation is redone with
% the second value, &c.
C_OR_range_BAL = 1 - (d_min:0.05:0.9);
C_OR_range_STOR = 1 - (d_min:0.05:0.9);

% [set by user] Threshold for determining whether HPP is "large" or "small" - if
% t_fill (eq. S1) is larger than threshold, classify as "large"
T_fill_thres = 1.0;

% [set by user] Optional: Requirement on Loss of Energy Expectation  (criterion (ii) on page 1 and Figure S1).
% As default, the HSW mix does not allow for any LOEE. However, this criterion could be relaxed.
% E.g. LOEE_allowed = 0.01 would mean that criterion (ii) is relaxed to 1% of yearly allowed LOEE instead of 0%.
LOEE_allowed = 0;

% [set by user] The parameter f_size is the percentile value described in eq. S11
f_size = 90;


%% pre.3) Static parameters

% [read] name of hydropower plant
HPP_name = ["Bui" "Buyo"];

% [read] relative capacity of solar and wind to be installed
c_solar_relative = [0.573210768220617 1];
c_wind_relative = 1 - c_solar_relative;

% [read] maximum head (m)
h_max = [80 36.1];

% [read] maximum lake area (m^2)
A_max = [4.4e8 9e8];

% [read] maximum storage volume (m^3)
V_max = [1.257e10 8.3e9];

% [read] turbine capacity (MW)
P_r_turb = [400 165];

% [set by user] if using STOR scenario: lower reservoir capacity (MW)
V_lower_max = V_max./10^3;

% [set by user] if using STOR scenario (only for Bui): pump capacity (MW)
P_r_pump = [100 NaN];

% [calculate] turbine and pump throughput (m^3/s, see explanation following eq. S8)
Q_max_turb = P_r_turb./(eta_turb*rho*g*h_max)*10^6;
Q_max_pump = P_r_pump./(eta_pump^(-1)*rho*g*h_max)*10^6;



%% pre.4) Time series

% [read] Load curves (L_norm; see eq. S10)
L_norm(:,:,1) = xlsread('minimum_example_load.xlsx','GH');
L_norm(:,:,2) = xlsread('minimum_example_load.xlsx','CIV');

% [read] Precipitation and evaporation flux (kg/m^2/s)
precipitation_flux_hourly(:,:,1) = xlsread('minimum_example_precipitation','Bui');
precipitation_flux_hourly(:,:,2) = xlsread('minimum_example_precipitation','Buyo');

evaporation_flux_hourly(:,:,1) = xlsread('minimum_example_evaporation','Bui');
evaporation_flux_hourly(:,:,2) = xlsread('minimum_example_evaporation','Buyo');

% [read] natural inflow at hourly timescale (m^3/s)
Q_in_nat_hourly(:,:,1) = xlsread('minimum_example_inflow.xlsx','Bui');
Q_in_nat_hourly(:,:,2) = xlsread('minimum_example_inflow.xlsx','Buyo');

% [read] capacity factors weighted by location (eq. S12)
CF_solar_hourly(:,:,1) = xlsread('minimum_example_CF_solar.xlsx','GH');
CF_solar_hourly(:,:,2) = xlsread('minimum_example_CF_solar.xlsx','CIV');

CF_wind_hourly(:,:,1) = xlsread('minimum_example_CF_wind.xlsx','GH');
CF_wind_hourly(:,:,2) = xlsread('minimum_example_CF_wind.xlsx','CIV');


%% pre.5) Bathymetry

% [read] Calibration curves used during simulations
temp = xlsread('minimum_example_bathymetry.xlsx','Bui');
% [extract] volume (m^3)
calibrate_volume(:,1) = temp(:,1);
% [extract] area (m^2)
calibrate_area(:,1) = temp(:,2);
% [extract] head (m)
calibrate_head(:,1) = temp(:,3);

% [read] Calibration curves used during simulations
temp = xlsread('minimum_example_bathymetry.xlsx','Buyo');
% [extract] volume (m^3)
calibrate_volume(:,2) = temp(:,1);
% [extract] area (m^2)
calibrate_area(:,2) = temp(:,2);
% [extract] head (m)
calibrate_head(:,2) = temp(:,3);

clear temp
