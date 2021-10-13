==============================================
Code for simulating Estonia's electricity grid
==============================================

Context
-------

Many countries have committed to reducing their carbon dioxide emissions in an effort to address the climate change caused by such greenhouse gases. Estonia, for instance, has committed to becoming a climate neutral economy by 2050_. The energy sector is the main emitter of greenhouse gases in Estonia, so to achieve its goal Estonia must replace existing oil shale power plants with low-emission alternatives.

To compare the different alternatives, we ran simulations to estimate the cost and stability of various potential energy systems that Estonia could transition to. This repository contains the code used for those simulations, and the results are discussed in detail in our article: `"Comparison of the most likely low-emission electricity production systems in Estonia"`_.

Here we also give an overview of the code to help the reader more quickly understand how the code works and what parameters were used.

Simulation
----------
The core of the code is the `simulation <https://github.com/zmeri/electricity-sim-estonia/blob/master/electric_grid_est.py#L21>`_ function, which performs a single year-long simulation of the electricity grid in Estonia with a specified capacity of various energy technologies. The function calls the demand, production, and storage functions to estimate electricity flows in the grid at a given time. A resolution of `10 seconds <https://github.com/zmeri/electricity-sim-estonia/blob/master/electric_grid_est.py#L25>`_ was used.

Demand
------

The electricity demand for a given year is generated using the `demand <https://github.com/zmeri/electricity-sim-estonia/blob/master/electric_grid_est.py#L277>`_ function. This demand data is generated using an inverse discrete cosine transform, which is implemented using Scipy's idct_ function. The cosine coefficients that are transformed are sampled from normal distributions. The parameters for these normal distributions were determined before based on 5 years of actual demand data for Estonia, and this was done using the `find_demand_coefficients <https://github.com/zmeri/electricity-sim-estonia/blob/master/electric_grid_est.py#L425>`_ function.

Production
----------

The amount of electricity produced is estimated using the `production <https://github.com/zmeri/electricity-sim-estonia/blob/master/electric_grid_est.py#L126>`_ function.

Storage
-------

The amount of electricity flowing to or from storage was calculated in a separate `C++ file <https://github.com/zmeri/electricity-sim-estonia/blob/master/storage_func/storage_base.cpp#L8>`_. This was done because the logic for calculating how much to send or take from storage was more complicated and required using a for loop that was too slow in Python. The C++ code was integrated into the remaining Python code using Cython.

Cost
----

The overall system cost is calculated in the `calc_cost <https://github.com/zmeri/electricity-sim-estonia/blob/master/electric_grid_est.py#L303>`_ function. We calculated the levelized cost using the equation used by IRENA_ and most other literature sources (see Equation 1 in `our article`_). For `biomass plants <https://github.com/zmeri/electricity-sim-estonia/blob/master/electric_grid_est.py#L341>`_, `hydropower <https://github.com/zmeri/electricity-sim-estonia/blob/master/electric_grid_est.py#L345>`_, and `onshore <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L151>`_ and `offshore <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L159>`_ wind turbines, we simply used distributions based on literature data. Below we give the main parameters used in calculating the levelized cost of electricity for each technology.

Oil shale
~~~~~~~~~

The levelized cost of oil shale power plants is estimated in the `cost_os <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L10>`_ function. Here are the main parameters used in the calculations:

*  CO\ :sub:`2` capture level = `0.9 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L14>`_
*  Mean CO\ :sub:`2` emission factor = `0.930 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L15>`_ tonne CO\ :sub:`2`\ /MWh
*  Mean operation, maintenance, and fuel cost = `20 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L16>`_ EUR/MWh
*  Mean cost of CO\ :sub:`2` credits = `40 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L18>`_ tonne CO\ :sub:`2`\ /MWh
*  Plant lifetime = `35 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L22>`_ years
*  Construction time = `4 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L23>`_ years
*  Mean overnight construction cost = `1.87e6 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L24>`_ EUR/MW
*  Mean cost of CCS = `63 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L27>`_ EUR/tonne CO\ :sub:`2`
*  Mean cost of CCS (next gen) = `37 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L28>`_ EUR/tonne CO\ :sub:`2`
*  Mean cost of CO\ :sub:`2` transport and storage = `25 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L26>`_ EUR/tonne CO\ :sub:`2`

Pyrolysis gas
~~~~~~~~~

The levelized cost of Pyrolysis gas power plants is estimated in the `cost_os_gas <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L48>`_ function. Here are the main parameters used in the calculations:

*  CO\ :sub:`2` capture level = `0.9 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L52>`_
*  Mean CO\ :sub:`2` emission factor = `0.2 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L53>`_ tonne CO\ :sub:`2`\ /MWh
*  Mean operation, maintenance, and fuel cost = `10 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L54>`_ EUR/MWh
*  Mean cost of CO\ :sub:`2` credits = `40 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L55>`_ tonne CO\ :sub:`2`\ /MWh
*  Plant lifetime = `35 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L57>`_ years
*  Construction time = `3 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L58>`_ years
*  Mean overnight construction cost = `1.6e6 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L59>`_ EUR/MW
*  Mean cost of CCS = `63 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L62>`_ EUR/tonne CO\ :sub:`2`
*  Mean cost of CCS (next gen) = `37 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L63>`_ EUR/tonne CO\ :sub:`2`
*  Mean cost of CO\ :sub:`2` transport and storage = `25 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L61>`_ EUR/tonne CO\ :sub:`2`

Solar
~~~~~

The levelized cost of solar panels is estimated in the `cost_solar <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L126>`_ function. Here are the main parameters used in the calculations:

*  Mean operating costs = `17.8 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L129>`_ EUR/MWh
*  Plant lifetime = `25 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L127>`_ years
*  Mean overnight construction cost = `1199.91 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L128>`_ EUR/MW
When simulating the cost of the grid as a whole, the additional costs of `grid improvements <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L134>`_ needed for renewables was also included, as has been described by `Ueckerdt et al. (2013)`_. The cost of grid improvements also added to `offshore wind <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L166>`_ in the simulations.

Nuclear
~~~~~~~

The levelized cost of a nuclear plant with a small modular reactor is estimated in the `cost_nuclear <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L178>`_ function. Here are the main parameters used in the calculations:

*  Mean operating and maintenance costs = `10 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L189>`_ EUR/MWh
*  Mean fuel costs (including disposal) = `6 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L187>`_ EUR/MWh
*  Plant lifetime = `60 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L182>`_ years
*  Construction time = `7 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L183>`_ years
*  Mean overnight construction cost = `4402.82 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L186>`_ EUR/MW

Underground pumped hydro storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Underground pumped hydro storage was the electricity storage technology selected for use in the simulations due to its low cost compared to other storage technologies and due to the maturity of the technology. The levelized cost of an underground pumped hydro facility is estimated in the `cost_storage_uphes <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L203>`_ function. Here are the main parameters used in the calculations:

*  Plant lifetime = `60 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L211>`_ years
*  Construction time = `8 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L212>`_ years
*  Mean overnight construction cost for conversion equipment (e.g. pumps and turbines) = `1200 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L218>`_ EUR/MW
*  Mean overnight construction cost for storage reservoir = `30 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L219>`_ EUR/MWh
*  Scaling factor for conversion equipment = `0.4 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L215>`_
*  Scaling factor for storage reservoir = `0.85 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L214>`_
*  Basis for scaling equation for conversion equipment = `500 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L217>`_ MW
*  Basis for scaling equation for storage reservoir = `6000 <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L216>`_ MWh
`Operating expenses <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L222>`_ were calculated based on literature estimates of the maintenance and labor costs. The amount of `potential revenue <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L225>`_ from selling electricity at a higher price than it cost to store it (arbitrage) was also included.

The cost_models.py file also contains functions for estimating the cost of `biomass plants <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L83>`_ and `hydrogen storage <https://github.com/zmeri/electricity-sim-estonia/blob/master/cost_models.py#L242>`_, although these were not used in the final analysis.

Monte Carlo method
------------------

For the analyses performed in this study, a `Monte Carlo <https://github.com/zmeri/electricity-sim-estonia/blob/master/electric_grid_est.py#L361>`_ method was used to run the simulations many times to capture the full range of variation caused by the uncertainty of the underlying assumptions and parameters.

Analyses
--------

Several functions were written to perform the following analyses for this study:

*  To investigate how `storage <https://github.com/zmeri/electricity-sim-estonia/blob/master/electric_grid_est.py#L505>`_ affects the performance of a grid consisting almost entirely of wind turbines
*  To `compare <https://github.com/zmeri/electricity-sim-estonia/blob/master/electric_grid_est.py#L571>`_ likely low-emission scenarios in Estonia
*  To determine how the cost and net surplus change with increasing `penetration <https://github.com/zmeri/electricity-sim-estonia/blob/master/electric_grid_est.py#L756>`_ of wind power
*  To calculate `surplus duration <https://github.com/zmeri/electricity-sim-estonia/blob/master/electric_grid_est.py#L653>`_ curves

References
----------

Various literature sources are also cited in the code to show where certain values were obtained. These references are given as IDs preceded by an @ symbol. The corresponding metadata for the reference can be looked up in the `references.json`_ file.

License
-------

This project is licensed under the GNU General Public License v3.0

.. _2050: https://www.riigiteataja.ee/akt/315052021012
.. _`"Comparison of the most likely low-emission electricity production systems in Estonia"`: https://osf.io/rq5kp
.. _idct: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.idct.html
.. _IRENA: https://www.irena.org/publications/2020/Jun/Renewable-Power-Costs-in-2019
.. _`our article`: https://osf.io/rq5kp
.. _`Ueckerdt et al. (2013)`: https://doi.org/10.1016/j.energy.2013.10.072
.. _`references.json`: https://github.com/zmeri/electricity-sim-estonia/references.json