
# REVUB (Renewable Electricity Variability, Upscaling and Balancing) 

# <img src="./graphs/header_logo.png" align="right" />

Authors: Sebastian Sterl


Contact author: sebastian.sterl@vub.be

# 1. Introduction
---
The main objective of REVUB is to model how the operation and dispatch of hydropower plants can be hybridised with variable solar and wind power (VRE) plants, allowing the combination of hydro with VRE to operate "as a single unit" to provide reliable electricity supply and load-following services. 

The model can be used, for instance, in due diligence processes for power plant financing.

This model has been used for several peer-reviewed publications related to regional power system interconnection, solar/wind integration, and drought resilience.

A detailed description of all involved principles and equations can be found in the dedicated [Manual](https://github.com/VUB-HYDR/REVUB/blob/master/3_Manual/REVUB_manual.pdf).

The REVUB code simulates dedicated hydropower plant operation to provide an effective capacity credit to VRE under technical, hydrological, and environmental constraints, and allows to derive:

* Suitable mixes of hydro, solar and wind power to maximise load-following under user-defined constraints;
* Operating rules for hydropower reservoirs to enable this load-following across wet- and dry-year conditions;
* Hourly to decadally resolved hydro, solar and wind power generation and dispatch.

# 2. Installation
---
The most recent version of the REVUB model was written for Python 3.13.

The [envs](https://github.com/VUB-HYDR/REVUB/tree/master/envs) folder contains a .yaml configuration file to install the needed packages to run the latest REVUB release.

A [training dataset](https://github.com/VUB-HYDR/REVUB/tree/master/4_Training), allowing the user to set up a REVUB simulation from scratch, learn how to set up input data, and become acquainted with simulation control, is available in English (EN) and Spanish (ES).

# 3. Tool's structure
---

### Scripts
The code is divided into [four scripts](https://github.com/VUB-HYDR/REVUB/tree/master/1_REVUB_code): one for initialisation (A), one containing the core code (B), and two for plotting (C). For a detailed explanation of the purpose of each file and the equations solved by the core code, the user is referred to the Manual. The files are always run in sequence A-B-C.

* **A_REVUB_initialise**

This script initialises the data needed for a simulation to run.

The script is controlled by a "control file" in Excel (*control_file.xlsx*) where the user controls overall modelling parameters, and reads in several Excel files with tabulated time series and other data (*data_xxx.xlsx*). 

In the training dataset, the user learns how to work with these files.

An [auxiliary script](https://github.com/VUB-HYDR/REVUB/tree/master/2_Auxiliary_script) is available to convert common non-hourly data formats into (pseudo)-hourly time series required for the *data_xxx.xlsx* files.

* **B_REVUB_main_code**

This script runs the actual REVUB model simulation and optimisation.

In the training dataset, the user learns how to run this code after having successfully initialised a simulation.
 
* **C_REVUB_plotting_individual**

This script produces figure and table outputs for the individually simulated plants, chosen by the user from the control file.

The figures include various time series and statistical charts on - among other things - reservoir dynamics (drawdown and refilling) without and with hydro-VRE hybridisation, electricity generation of the hydro-VRE complex from hourly to seasonal and multianual scales, and the corresponding hydropower plant operation (rule curves, turbine activity, mode of operation).

In the training dataset, the user learns how to produce meaningful figures using this script after having successfully run a simulation.


* **C_REVUB_plotting_multiple**

This script produces figure and table outputs of the overall power mix of a given region/country/grid. 

For a user-defined ensemble of the simulated plants, which the user can set in the control file, the script aggregates overall hydro-solar-wind power generation from this ensemble at hourly, seasonal and multiannual time scales, and compares it to a user-set overall hourly power demand curve (representing overall demand in the country/region/grid). 

The difference between hydro-VRE and this overall demand is assumed to be covered by other flexible power sources (e.g. thermal power sources or imports). Thus, this script can be used to provide insights on the overall power mix of a country/region/grid upon implementing hydro-VRE complementary operation.

The script also provides statistics on ramping needs of hydropower plants and other flexible sources under the investigated scenarios.

## Versions
Version 1.2.0 - July 2026

Version 1.1.1 - January 2026

Version 1.1.0 - May 2025

Version 1.0.4 - April 2024

Version 1.0.3 - November 2023

Version 1.0.2 - October 2023

Version 1.0.1 - September 2023

Version 1.0.0 - August 2023

Version 0.1.0 - January 2020

## License
See also the [LICENSE](./LICENSE.md) file.

## Acknowledgements
The following funding sources are gratefully acknowledged:
* 2025-26: Flemish Inter-University Development Cooperation ([VLIR-UOS](https://www.vliruos.be/)), Global Minds XREI Staff Mobility project "Ecuador's electricity crisis 2023-24: Drought impacts on hydropower and mitigation through variable renewables". 
* 2023-24: [The World Bank](https://www.worldbank.org/ext/en/home) (WB), Infrastructure Energy & Extractives Global Knowledge Unit (IEEGK), PO Number 8831242.
* 2022: [International Rivers](https://www.internationalrivers.org/), consulting contract "Alternatives to new hydropower to meet energy needs in the Republic of Guinea".
* 2018-21: Project CIREG, part of [ERA4CS](https://jpi-climate.eu/programme/era4cs/), an ERA-NET Co-fund action initiated by JPI Climate, funded by BMBF (DE), FORMAS (SE), BELSPO (BE) and IFD (DK) with co-funding from the EU's Horizon2020 Framework Program, Grant 690462.

## References
The REVUB model has so far been used in, and/or inspired the methods of, the following publications/documents:
* S. Sterl, L.E. Pineda, T. Mast, Y. Rodríguez, P. Muñoz, and W. Thiery. <ins>_[Variable renewables fortify Ecuador's power system against recurrences of drought-driven energy crises](https://doi.org/10.1038/s44221-026-00617-w)_</ins>. Nature Water __4__, 571–585 (2026).
* H. Hoff, M. Ogeya, D. de Condappa, R.J. Brecha, M.A.D. Larsen, K. Halsnæs, S. Salack, S. Sanfo, S. Sterl, and S. Liersch. <ins>_[Stakeholder-guided, model-based scenarios for a climate- and water-smart electricity transition in Ghana and Burkina Faso](https://doi.org/10.1016/j.esr.2023.101149)_</ins>. Energy Strategy Reviews __49__ (2023).
* P. Donk, S. Sterl, W. Thiery, and P. Willems. <ins>_[A policy framework for power system planning towards optimized integration of renewables under potential climate change – The Small Island Developing States perspective](https://doi.org/10.1016/j.enpol.2023.113526)_</ins>. Energy Policy __177__ (2023).
* S. Sterl and W. Thiery. <ins>_[La faisabilité du solaire PV pour remplacer la centrale hydroélectrique de Koukoutamba en Guinée: Étude quantitative](http://dx.doi.org/10.13140/RG.2.2.26548.83848)_</ins>. Vrije Universiteit Brussel, Brussels, Belgium (2022).
* S. Sterl, A. Devillers, C.J. Chawanda, A. van Griensven, W. Thiery, and D. Russo. <ins>_[A spatiotemporal atlas of hydropower in Africa for energy modelling purposes](https://doi.org/10.12688/openreseurope.13392.3)_</ins>. Open Research Europe __1__, 29 (2021).
* S. Sterl, D. Fadly, S. Liersch, H. Koch, and W. Thiery. <ins>_[Linking solar and wind power in eastern Africa with operation of the Grand Ethiopian Renaissance Dam](https://doi.org/10.1038/s41560-021-00799-5)_</ins>. Nature Energy __6__, 407–418 (2021).
* P. Donk, S. Sterl, W. Thiery, and P. Willems. <ins>_[REVUB-Light: A parsimonious model to assess power system balancing and flexibility for optimal intermittent renewable energy integration—A study of Suriname](https://doi.org/10.1016/j.renene.2021.03.117)_</ins>. Renewable Energy __173__, 57–75 (2021).
* S. Sterl, P. Donk, P. Willems, and W. Thiery. <ins>_[Turbines of the Caribbean: Decarbonising Suriname’s electricity mix through hydro-supported integration of wind power](https://doi.org/10.1016/j.rser.2020.110352)_</ins>. Renewable and Sustainable Energy Reviews __134__ (2020) 110352.
* S. Sterl, I. Vanderkelen, C.J. Chawanda, D. Russo, R.J. Brecha, A. van Griensven, N.P.M. van Lipzig, and W. Thiery. <ins>_[Smart renewable electricity portfolios in West Africa](https://doi.org/10.1038/s41893-020-0539-0)_</ins>. Nature Sustainability __3__, 710–719 (2020).