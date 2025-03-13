
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
The most recent version of the REVUB model was written for Python 3.9.

No specific packages are needed except for the regular numpy, pandas, and matplotlib.

A training dataset, allowing the user to set up a REVUB simulation from scratch, learn how to set up input data, and become acquainted with simulation control, is available in the folder https://github.com/VUB-HYDR/REVUB/tree/master/5_Training_dataset.

# 3. Tool's structure
---

### Scripts
The code is divided into four scripts: one for initialisation (A), one containing the core code (B), and two for plotting (C). For a detailed explanation of the purpose of each file and the equations solved by the core code, the user is referred to the Manual in the folder https://github.com/VUB-HYDR/REVUB/tree/master/4_Manual. The files are always run in sequence A-B-C.

* **A_REVUB_initialise**

This script initialises the data needed for a simulation to run.

The script is controlled by an Excel file where the user defines overall modelling parameters ("parameters_simulation.xlsx"), and reads in several Excel files with tabulated time series and other data ("data_xxx.xlsx"). 

In the training dataset, the user learns how to work with these files. 

* **B_REVUB_main_code**

This script runs the actual REVUB model simulation and optimisation.

In the training dataset, the user learns how to run this code after having successfully initialised a simulation.
 
* **C_REVUB_plotting_individual**

This script produces figure outputs for the individually simulated plants, chosen by the user from an Excel file named "plotting_settings.xlsx".

The figures include various time series and statistical charts on - among other things - reservoir dynamics (drawdown and refilling) without and with hydro-VRE hybridisation, electricity generation of the hydro-VRE complex from hourly to seasonal and multianual scales, and the corresponding hydropower plant operation (rule curves, turbine activity, mode of operation).

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

## Acknowledgements
The following funding sources are gratefully acknowledged:
* 2023-24: The World Bank (WB), Infrastructure Energy & Extractives Global Knowledge Unit (IEEGK), PO Number 8831242.
* 2022: International Rivers, consulting contract "Alternatives to new hydropower to meet energy needs in the Republic of Guinea".
* 2018-21: Project CIREG, part of ERA4CS, an ERA-NET Co-fund action initiated by JPI Climate, funded by BMBF (DE), FORMAS (SE), BELSPO (BE) and IFD (DK) with co-funding from the EU's Horizon2020 Framework Program, Grant 690462.

## References
The REVUB model has so far been used in, and/or inspired the methods of, the following publications/documents:
* S. Sterl, I. Vanderkelen, C.J. Chawanda, D. Russo, R.J. Brecha, A. van Griensven, N.P.M. van Lipzig, and W. Thiery. <ins>_Smart renewable electricity portfolios in West Africa_</ins>. Nature Sustainability __3__, 710–719 (2020). https://doi.org/10.1038/s41893-020-0539-0.
* S. Sterl, P. Donk, P. Willems, and W. Thiery. <ins>_Turbines of the Caribbean: Decarbonising Suriname’s electricity mix through hydro-supported integration of wind power_</ins>. Renewable and Sustainable Energy Reviews __134__ (2020) 110352. https://doi.org/10.1016/j.rser.2020.110352.
* P. Donk, S. Sterl, W. Thiery, and P. Willems. <ins>_REVUB-Light: A parsimonious model to assess power system balancing and flexibility for optimal intermittent renewable energy integration—A study of Suriname_</ins>. Renewable Energy __173__, 57–75 (2021). https://doi.org/10.1016/j.renene.2021.03.117.
* S. Sterl, D. Fadly, S. Liersch, H. Koch, and W. Thiery. <ins>_Linking solar and wind power in eastern Africa with operation of the Grand Ethiopian Renaissance Dam_</ins>. Nature Energy __6__, 407–418 (2021). https://doi.org/10.1038/s41560-021-00799-5.
* S. Sterl, A. Devillers, C.J. Chawanda, A. van Griensven, W. Thiery, and D. Russo. <ins>_A spatiotemporal atlas of hydropower in Africa for energy modelling purposes_</ins>. Open Research Europe __1__, 29 (2021). https://doi.org/10.12688/openreseurope.13392.3.
* S. Sterl and W. Thiery. <ins>_La faisabilité du solaire PV pour remplacer la centrale hydroélectrique de Koukoutamba en Guinée: Étude quantitative_</ins>. Vrije Universiteit Brussel, Brussels, Belgium (2022). http://dx.doi.org/10.13140/RG.2.2.26548.83848.
* P. Donk, S. Sterl, W. Thiery, and P. Willems. <ins>_A policy framework for power system planning towards optimized integration of renewables under potential climate change – The Small Island Developing States perspective_</ins>. Energy Policy __177__ (2023). https://doi.org/10.1016/j.enpol.2023.113526.
* H. Hoff, M. Ogeya, D. de Condappa, R.J. Brecha, M.A.D. Larsen, K. Halsnæs, S. Salack, S. Sanfo, S. Sterl, and S. Liersch. <ins>_Stakeholder-guided, model-based scenarios for a climate- and water-smart electricity transition in Ghana and Burkina Faso_</ins>. Energy Strategy Reviews __49__ (2023). https://doi.org/10.1016/j.esr.2023.101149.
