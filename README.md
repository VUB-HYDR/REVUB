
# REVUB (Renewable Electricity Variability, Upscaling and Balancing) 

# <img src="./graphs/header_logo.png" align="right" />

Authors: Sebastian Sterl


Contact author: sebastian.sterl@vub.be

# 1. Introduction
---
The main objective of REVUB is to model how the operation of hydropower plants can be hybridised with variable solar and wind power (VRE) plants, allowing the combination of hydro with VRE to operate "as a single unit" to provide reliable electricity supply and load-following services. The model can be used, for instance, in due diligence processes for power plant financing.

This model was first introduced in the paper "Smart renewable electricity portfolios in West Africa" by Sterl et al. (2020; https://www.nature.com/articles/s41893-020-0539-0); hereafter referred to as "the publication". It has since been used for several more peer-reviewed publications.

A detailed description of all involved principles and equations can be found in the dedicated Manual (https://github.com/VUB-HYDR/REVUB/blob/master/4_Manual/REVUB_manual.pdf).

The REVUB code simulates dedicated hydropower plant operation to provide an effective capacity credit to VRE, and allows to derive:

* Suitable mixes of hydro, solar and wind power to maximise load-following under user-defined constraints;
* Reliable operating rules for hydropower reservoirs to enable this load-following across wet- and dry-year conditions;
* Hourly to decadally resolved hydro, solar and wind power generation.

# 2. Installation
---
The most recent version of the REVUB model was written for Python 3.9. The files given in this GitHub folder contain code and data needed to run a minimum working example. 

No specific packages are needed except for the regular numpy, pandas, and matplotlib.

# 3. Tool's structure
---

### Scripts
The code is divided into four scripts: one for initialisation (A), one containing the core code (B), and two for plotting (C). For a detailed explanation of the purpose of each file, the user is referred to the Manual in the folder https://github.com/VUB-HYDR/REVUB/tree/master/4_Manual. The files are always run in sequence A-B-C.

A training dataset, allowing the user to set up a REVUB simulation from scratch, learn how to set up input data, and become acquainted with simulation control, is available in the folder https://github.com/VUB-HYDR/REVUB/tree/master/5_Training_dataset.

* **A_REVUB_initialise**

This script initialises the data needed for a simulation to run.

The script is controlled by an Excel file where the user defines overall modelling parameters ("parameters_simulation.xlsx"), and reads in several Excel files with tabulated time series and other data ("data_xxx.xlsx"). 

In the training dataset, the user learns how to work with these files. 

* **B_REVUB_main_code**

This script runs the actual REVUB model simulation and optimisation.

In the training dataset, the user learns how to run this code after having successfully initialised a simulation.
 
* **C_REVUB_plotting_individual**

This script produces figure outputs for the individually simulated plants, chosen by the user from an Excel file named "plotting_settings.xlsx".

The figures include (i) time series of hydropower lake levels and reservoir outflows without and with complementary hydro-VRE operation, (ii) power generation curves from the selected hydropower plant alongside solar and wind power at hourly, seasonal and multiannual scales, and (iii) hydropower release rule curves for given months and times of the day.

In the training dataset, the user learns how to produce meaningful figures using this script after having successfully run a simulation.


* **C_REVUB_plotting_multiple**

This script produces figure outputs of the overall power mix of a given region/country/grid. 

For a user-defined ensemble of the simulated plants, which the user can set in the Excel file "plotting_settings.xlsx", the script plots overall hydro-solar-wind power generation from this ensemble at hourly, seasonal and multiannual time scales, and compares it to a user-set overall hourly power demand curve (representing overall demand in the country/region/grid). 

The difference between hydro-VRE and this overall demand is assumed to be covered by other power sources (thermal power sources are used as default in the script). Thus, this script can be used to provide insights on the overall power mix of a country/region/grid upon implementing hydro-VRE complementary operation.

## Versions
Version 0.1.0 - January 2020

Version 1.0.0 - August 2023

Version 1.0.1 - September 2023

Version 1.0.2 - October 2023

Version 1.0.3 - November 2023

Version 1.0.4 - April 2024

## License
See also the [LICENSE](./LICENSE.md) file.

