
# REVUB (Renewable Electricity Variability, Upscaling and Balancing) 

# <img src="./graphs/header_logo.png" align="right" />

Authors: Sebastian Sterl


Contact author: sebastian.sterl@vub.be

# 1. Introduction
---
The main objective of REVUB is to model how flexible operation of hydropower plants can help renewable electricity mixes with variable solar and wind power to provide reliable electricity supply and load-following services.
This model was first introduced in the paper "Smart renewable portfolios to displace fossil fuels and avoid hydropower overexploitation" (Sterl et al., in preparation); hereafter referred to as "the publication".
A detailed description of all involved principles and equations can be found in the publication and its SI.

The REVUB code models reservoir operation and hydro-solar-wind power generation and derives:

* Optimal mix of hydro / solar / wind to maximise load-following;
* Optimal operating rules for hydropower reservoirs to enable this load-following;
* Hourly to decadally resolved hydro, solar and wind power generation.

# 2. Installation
---
To run the model, the user should be able to operate in a MATLAB environment.
The code was written for MATLAB R2017b.
The files given in this GitHub folder contain code and data needed to run a minimum working example.

# 3. Tool's structure
---

### Scripts
There are three main scripts:
* **A_REVUB_initialise_minimum_example.m**

The first script initialises the data needed for the minimum working example to run (which covers Bui hydropower plant in Ghana, and Buyo hydropower plant in Côte d'Ivoire). It reads in several time series from Excel sheets (these time series are themselves the results of external computations, described in the publication). These datasets are given in the folder "data" (extract the archive "data.rar"). The extracted data files should be in the same folder in which this script is located.
* **B_REVUB_main_code.m**

The second script runs the actual REVUB model simulation and optimisation.
 
* **C_REVUB_plotting.m**. 

The third script produces figure outputs for Bui (set plot_HPP = 1) or Buyo (set plot_HPP = 2), most of which can also be found in the publication or its SI (for the same example).

To produce the figure outputs, simply run the scripts in order of numbering. The second script may take several minutes to run.

## Versions
Version 0.1.0 - September 2019  

## License
See also the [LICENSE](./LICENSE.md) file.

