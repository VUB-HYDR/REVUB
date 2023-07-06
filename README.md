
# REVUB (Renewable Electricity Variability, Upscaling and Balancing) 

# <img src="./graphs/header_logo.png" align="right" />

Authors: Sebastian Sterl


Contact author: sebastian.sterl@vub.be

# 1. Introduction
---
The main objective of REVUB is to model how flexible operation of hydropower plants can help renewable electricity mixes with variable solar and wind power to provide reliable electricity supply and load-following services.
This model was first introduced in the paper "Smart renewable electricity portfolios in West Africa" by Sterl et al. (2020; https://www.nature.com/articles/s41893-020-0539-0); hereafter referred to as "the publication".
A detailed description of all involved principles and equations can be found in the publication and its SI, as well as in the Manual (https://github.com/VUB-HYDR/REVUB/blob/master/manual/REVUB_manual.pdf).

The REVUB code models reservoir operation and hydro-solar-wind power generation and derives:

* Optimal mix of hydro / solar / wind to maximise load-following;
* Optimal operating rules for hydropower reservoirs to enable this load-following;
* Hourly to decadally resolved hydro, solar and wind power generation.

# 2. Installation
---
The model exists in two languages: a Python version (written for Python 3.9) and a MATLAB version (written for MATLAB R2017b). It is recommended to use the Python version, as the MATLAB version is no longer being supported by code updates.
The files given in this GitHub folder contain code and data needed to run the same minimum working example for both languages. 

# 3. Tool's structure
---

### Scripts
There are four main scripts: one for initialisation (A), one containing the core code (B), and two for plotting (C). For a detailed explanation of the purpose of each file, the user is referred to the Manual.

* **A_REVUB_initialise_minimum_example**

This script initialises the data needed for the minimum working example to run (which covers Bui hydropower plant in Ghana, and Buyo hydropower plant in Côte d'Ivoire). It reads in an Excel file with overall modelling parameters, and several Excel files with tabulated time series (these time series are themselves the results of external computations, described in the publication). These datasets are given in the folder "data" (extract the archive "data.rar"). The extracted data files should be in the same folder in which this script is located. The names of the hydropower plants in the parameter sheet must match exactly the names of the worksheets of the spreadsheets containing the corresponding time series tables.

* **B_REVUB_main_code**

This script runs the actual REVUB model simulation and optimisation.
 
* **C_REVUB_plotting_individual**

This script produces figure outputs for the individually simulated plants, in this case Bui or Buyo, most of which can also be found in the publication or its SI (for the same example).

* **C_REVUB_plotting_multiple**

This script produces figure outputs of the overall power mix of a given region/country/grid. For a user-defined ensemble of the simulated plants (in this minimum example, the options for this ensemble are (i) Bui, (ii) Buyo, and (iii) Bui + Buyo together), the script plots overall hydro-solar-wind power generation and compares it to a user-set overall power demand curve (representing overall demand in the country/region/grid). The difference between hydro-VRE and this overall demand is assumed to be covered by other power sources (thermal power sources are used as default in the script). Thus, this script can be used to provide insights on the overall power mix of a country/region/grid upon implementing hydro-VRE complementary operation.

To produce the figure outputs, simply run the scripts (.py for Python, .m for MATLAB) in the order A-B-C.

## Versions
Version 0.1.0 - January 2020

Version 0.1.1 - July 2023

## License
See also the [LICENSE](./LICENSE.md) file.

