# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:37:38 2020

@author: ssterl
"""

import numpy as np
import pandas as pd
import numbers as nb

# import data from Excel
filename_data = 'rearrange_data_template.xlsx'
data = pd.read_excel (filename_data, sheet_name = 'daily_bymonth_series', header = None)
data_list = np.array(data[0][0:].tolist())
data_values = np.array(data)[0:,1:]

# [remove] deactivated data series
columns_active = data_values[np.where(data_list == 'activate for conversion', True, False)][0]
data_values = np.delete(data_values, np.where(columns_active == 0)[0],1)
series_number = len(data_values[0,:])

# [extract] start and end year of user-entered monthly series
first_year = data_values[np.where(data_list == 'first_year', True, False)][0]
last_year = data_values[np.where(data_list == 'last_year', True, False)][0]

# [extract] data series headers
header = data_values[np.where(data_list == 'data series name', True, False)][0].tolist()

# [extract] data series to be converted from monthly to hourly
data_for_conversion = data_values[4:,:]

# [set] number of days in each month
dayhrs = 24
months_yr = 12
days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
days_leap = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
hours_month = dayhrs*days
hours_month_leap = dayhrs*days_leap
no_years = last_year[0] - first_year[0] + 1

# [preallocate] matrix for converted output at hourly scale
output_hourly_byyear = np.full([np.sum(days_leap)*dayhrs, no_years, series_number], np.nan)

# [loop] over the selected time series to parse monthly to hourly

for q in range(series_number):
    
    print('parsing data for', header[q])
    
    # [extract] first and last year
    year_start = first_year[q]
    year_end = last_year[q]
    no_years = year_end - year_start + 1
     
    # preallocate: positions of hours by month
    positions_month = np.zeros(len(days)+1)
    positions_month_leap = np.zeros(len(days)+1)
    for n in range(len(days) + 1):
        positions_month[n] = sum(hours_month[0:n])
        positions_month_leap[n] = sum(hours_month_leap[0:n])
        
    # write every value 24 times per day
    for y in range(no_years):
        year = year_start + y
        
        if (year/4).is_integer():
            for m in range(months_yr):
                output_hourly_byyear[int(positions_month_leap[m]) : int(positions_month_leap[m+1]), y, q] = np.tile(data_for_conversion[m*dayhrs : (m+1)*dayhrs, q], days_leap[m])
        else:
            for m in range(months_yr):
                output_hourly_byyear[int(positions_month[m]) : int(positions_month[m+1]), y, q] = np.tile(data_for_conversion[m*dayhrs : (m+1)*dayhrs, q], days[m])
    
