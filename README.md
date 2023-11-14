
# REVUB (Renewable Electricity Variability, Upscaling and Balancing) 

# <img src="./graphs/header_logo.png" align="right" />

Authors: Sebastian Sterl


Contact author: sebastian.sterl@vub.be

# 1. Introduction
---
The main objective of REVUB is to model how the operation of hydropower plants can be hybridised with variable solar and wind power (VRE) plants, allowing the combination of hydro with VRE to operate "as a single unit" to provide reliable electricity supply and load-following services. The model can be used, for instance, in due diligence processes for power plant financing.

This model was first introduced in the paper "Smart renewable electricity portfolios in West Africa" by Sterl et al. (2020; https://www.nature.com/articles/s41893-020-0539-0); hereafter referred to as "the publication". It has since been used for several more peer-reviewed publications.

A detailed description of all involved principles and equations can be found in the dedicated Manual (https://github.com/VUB-HYDR/REVUB/blob/master/manual/REVUB_manual.pdf).

The REVUB code simulates dedicated hydropower plant operation to provide an effective capacity credit to VRE, and allows to derive:

* Suitable mixes of hydro, solar and wind power to maximise load-following under user-defined constraints;
* Reliable operating rules for hydropower reservoirs to enable this load-following across wet- and dry-year conditions;
* Hourly to decadally resolved hydro, solar and wind power generation.

# 2. Installation
---
The most recent version of the REVUB model was written for Python 3.9. The files given in this GitHub folder contain code and data needed to run a minimum working example. 

In the past, a MATLAB version (written for MATLAB R2017b) of the REVUB model existed, which can be obtained upon request but is no longer being supported by code updates.

# 3. Tool's structure
---

### Scripts
The code is divided into four scripts: one for initialisation (A), one containing the core code (B), and two for plotting (C). For a detailed explanation of the purpose of each file, the user is referred to the Manual.

* **A_REVUB_initialise_minimum_example**

This script initialises the data needed for the minimum working example to run (which covers Bui hydropower plant in Ghana, and Buyo hydropower plant in Côte d'Ivoire). 

It reads in an Excel file with overall modelling parameters ("parameters_simulation.xlsx"), and several Excel files with tabulated time series and other data ("data_xxx.xlsx"; in this case, these datasets are themselves the results of external computations, described in the publication). 

These datasets are given in the folder "data". These data files should be downloaded and placed in the same folder in which this script is located. The names of the worksheets of all files named "data_xxx.xlsx" must be linked to the corresponding hydropower plant with the parameters "HPP_name_data_xxx" in the file "parameters_simulation.xlsx".

The folder "data" contains a sub-folder with two auxiliary scripts to parse time series at the monthly scale or at the "typical day" scale per month to full hourly scale, the latter being the needed timescale for REVUB simulations. The conversion from monthly scale to full hourly scale is relevant for quantities such as inflow, evaporation flux and precipitation flux which may often only be available at monthly scale and for which the hourly detail is of limited importance for reservoir operation. The conversion from "typical day" scale per month to full hourly scale is relevant for quantities for which the generic day/night and seasonal dynamics are known, but full hourly details are missing, such as downstream irrigation requirements or, in some cases, simplified solar/wind capacity factor time series.

* **B_REVUB_main_code**

This script runs the actual REVUB model simulation and optimisation.
 
* **C_REVUB_plotting_individual**

This script produces figure outputs for the individually simulated plants, in this case Bui or Buyo, chosen by the user from an Excel file named "plotting_settings.xlsx". Most of these figures also be found in the publication or its SI (for the same example). 

The figures include (i) time series of hydropower lake levels and reservoir outflows without and with complementary hydro-VRE operation, (ii) power generation curves from the selected hydropower plant alongside solar and wind power at hourly, seasonal and multiannual scales, and (iii) hydropower release rule curves for given months and times of the day.

* **C_REVUB_plotting_multiple**

This script produces figure outputs of the overall power mix of a given region/country/grid. 

For a user-defined ensemble of the simulated plants, which the user can set in the Excel file "plotting_settings.xlsx" (in this minimum example, the options for this ensemble are (i) only Bui, (ii) only Buyo, and (iii) Bui + Buyo together), the script plots overall hydro-solar-wind power generation from this ensemble at hourly, seasonal and multiannual time scales, and compares it to a user-set overall hourly power demand curve (representing overall demand in the country/region/grid). 

The difference between hydro-VRE and this overall demand is assumed to be covered by other power sources (thermal power sources are used as default in the script). Thus, this script can be used to provide insights on the overall power mix of a country/region/grid upon implementing hydro-VRE complementary operation.

To produce the figure outputs, simply run the scripts in the order A-B-C.

## Versions
Version 0.1.0 - January 2020

Version 1.0.0 - August 2023

Version 1.0.1 - September 2023

Version 1.0.2 - October 2023

Version 1.0.3 - November 2023

## License
See also the [LICENSE](./LICENSE.md) file.

