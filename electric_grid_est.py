import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.fft import idct, dct
from scipy import interpolate
from scipy.stats import linregress
import json
import multiprocessing
from textwrap import wrap
import os.path
from storage_func import storage
import cost_models as co

plt.rcParams.update({'font.size': 13})

class SimulationError(Exception):
    def __init__(self, message):
        self.message = message

def simulation(caps, variable=None, show_plot=False, plot_example=False):
    rng = np.random.default_rng()
    SECONDS_IN_YR = 31536000
    SECONDS_IN_DAY = 86400
    time_step = 10 # seconds, ##param:time_step
    t = np.arange(0, SECONDS_IN_YR / time_step) * time_step # current second in the year
    charge_eff = 0.88 # for pumped hydro @kusakanaFeasibilityAnalysisRiver2015
    discharge_eff = 0.88 # for pumped hydro
    # charge_eff = 0.53 # for hydrogen; @sanghaiTechnoEconomicAnalysisHydrogen2013
    # discharge_eff = 0.40 # for hydrogen; based on estimates given in @sanghaiTechnoEconomicAnalysisHydrogen2013

    demanded = demand(t, rng, variable)
    produced, annual_prod = production(t, caps, demanded, rng, variable=variable, plot_example=plot_example)
    price = calc_price(t, rng, variable)

    y2 = dct(produced)
    y2 = y2[:25000]
    smoothed_produced = idct(y2, n=t.shape[0]) # we somewhat smooth the production here to avoid very rapid swings in pumped hydro storage power that would likely not correspond to actual operation.
    to_storage = storage(demanded, smoothed_produced, caps['storage_power'], caps['storage_energy'], charge_eff, discharge_eff, time_step)

    surplus = produced - demanded - to_storage
    exported = export(surplus, demanded)
    unused = surplus - exported
    unused[np.where(unused < 0)] = 0
    surplus = surplus - unused

    if show_plot:
        plt.figure(figsize=(10,7))

        years = ['2020', '2019', '2018', '2017', '2016']
        for yr in years:
            comp = pd.read_csv('data/electricity-production and consumption_{}.csv'.format(yr), delimiter=';')
            plt.plot((comp['Ajatempel (UTC)'] - comp['Ajatempel (UTC)'].iloc[0]) / SECONDS_IN_DAY, comp['Tarbimine'], alpha=0.5, label='Actual demand {}'.format(yr))

        # plt.plot(t / SECONDS_IN_DAY, price, color='purple', label='Price')
        plt.plot(t / SECONDS_IN_DAY, demanded, color='black', label='Simulated demand')
        # plt.plot(t / SECONDS_IN_DAY, produced, color='blue', label='Production')
        # plt.plot(t / SECONDS_IN_DAY, to_storage, color='green', label='To storage')
        plt.xlabel("Day of the year")
        plt.ylabel("Electricity demand (MW)")
        plt.xlim([232, 239])
        plt.ylim([500, 1150])
        plt.subplots_adjust(right=0.7)
        plt.legend(frameon=False, bbox_to_anchor=(1.5, 0.5), loc='center right')
        # plt.legend(frameon=False, loc='lower left')
        # plt.savefig("figures/demand_simulation.png", dpi=400)
        plt.show()

    # calculate metrics
    total_production = np.sum(produced) * time_step / 3600 # MWh
    total_export = np.sum(exported[np.where(exported > 0)]) * time_step / 3600 # MWh
    total_shortage = np.sum(surplus[np.where(surplus < 0)]) * time_step / 3600 # MWh
    total_unused = np.sum(unused) * time_step / 3600 # MWh
    net_surplus = np.sum(surplus) * time_step / 3600 # MWh

    # update annual production to account for unused electricity that is lost
    fraction_lost = total_unused / total_production
    for k, v in annual_prod.items():
        annual_prod[k] = v * (1 - fraction_lost)

    cost, dif_from_arbitrage = calc_cost(caps, annual_prod, fraction_lost, 'uphes', rng, price, to_storage, variable=variable)
    avg_cost = np.mean(cost)
    dif_from_arbitrage = np.mean(dif_from_arbitrage)

    return total_production, total_export, total_shortage, total_unused, net_surplus, avg_cost, dif_from_arbitrage

def probability_dist(data):
    probabilities = np.linspace(0, 100, 200)
    values = np.zeros_like(probabilities)
    for i, p in enumerate(probabilities):
        values[i] = np.percentile(data, p)
    return probabilities, values

def surplus_duration(caps, i):
    rng = np.random.default_rng()
    SECONDS_IN_YR = 31536000
    SECONDS_IN_DAY = 86400
    time_step = 10 # seconds
    t = np.arange(0, SECONDS_IN_YR / time_step) * time_step # current second in the year
    charge_eff = 0.88 # for pumped hydro @kusakanaFeasibilityAnalysisRiver2015
    discharge_eff = 0.88 # for pumped hydro

    demanded = demand(t, rng)
    produced, annual_prod = production(t, caps, demanded, rng)

    y2 = dct(produced)
    y2 = y2[:25000]
    smoothed_produced = idct(y2, n=t.shape[0]) # we somewhat smooth the production here to avoid very rapid swings in pumped hydro storage power that would likely not correspond to actual operation.
    to_storage = storage(demanded, smoothed_produced, caps['storage_power'], caps['storage_energy'], charge_eff, discharge_eff, time_step)

    surplus = produced - demanded - to_storage
    probs, surplus = probability_dist(surplus)
    return i * np.ones_like(probs), probs, surplus

def demand_duration(i):
    rng = np.random.default_rng()
    SECONDS_IN_YR = 31536000
    time_step = 10 # seconds
    t = np.arange(0, SECONDS_IN_YR / time_step) * time_step # current second in the year

    demanded = demand(t, rng)

    probs, demanded = probability_dist(demanded)
    return i * np.ones_like(probs), probs, demanded

def production(t, caps, demanded, rng, variable=None, plot_example=False):
    npts = t.shape[0]
    SECONDS_IN_YR = 31536000
    SECONDS_IN_DAY = 86400
    time_step = SECONDS_IN_YR / npts

    # calculate smoothed demand
    y3 = dct(demanded)
    y3 = y3[:1000]
    smoothed_demand = idct(y3, n=npts)

    # biomass CHP
    if not variable or variable == 'biomass':
        biomass = rng.normal(160, 20, 1) * np.ones((npts,)) # MW, @eestistatistikaametKE033ElektrijaamadeToodang
    else:
        biomass = rng.normal(160, 0, 1) * np.ones((npts,))

    # hydropower
    if not variable:
        hydro = rng.normal(6, 1.1, 1) # MW, @eestistatistikaametKE032ElektrijaamadeVoimsus
    else:
        hydro = rng.normal(6, 0, 1) # MW, @eestistatistikaametKE032ElektrijaamadeVoimsus

    # # solar
    if not variable or variable == 'capfac_solar':
        y2 = rng.normal(0, 6e4, 4000)
        y2[0] = rng.normal(4e5, 4e4, 1) # @theworldbankgroupGlobalSolarAtlas
        y2[2] = rng.normal(-9e6, 4e5, 1) # @ilmateenistusPaikesekiirguseAtlas
        y2[730] = -2e7
        weather = np.clip(rng.normal(0.5, 0.3, 365), 0.03, 1) # @scienceeducationresourcecenteratcarletoncollegePartExamineData
        weather = np.repeat(weather, SECONDS_IN_DAY / time_step)
        capfac_solar = idct(y2, n=npts) / time_step
    else:
        y2 = rng.normal(0, 0, 4000)
        y2[0] = rng.normal(4e5, 0, 1) # @theworldbankgroupGlobalSolarAtlas
        y2[2] = rng.normal(-9e6, 0, 1) # @ilmateenistusPaikesekiirguseAtlas
        y2[730] = -2e7
        weather = np.clip(rng.normal(0.5, 0, 365), 0.03, 1) # @scienceeducationresourcecenteratcarletoncollegePartExamineData
        weather = np.repeat(weather, SECONDS_IN_DAY / time_step)
        capfac_solar = idct(y2, n=npts) / time_step

    capfac_solar[np.where(capfac_solar < 0)] = 0
    capfac_solar[np.where(capfac_solar > 1)] = 1
    capfac_solar = weather * capfac_solar
    solar = capfac_solar * caps['solar']

    # wind turbines
    ncoef = 50000
    coef = pd.read_csv('data/wind_dct_coefficients.csv', delimiter=';') # loading the parameters for the coefficients
    coef = coef.head(ncoef)

    if not variable or variable == 'capfac_wind':
        y2 = rng.normal(coef['mean'], coef['stdev'], ncoef)
    else:
        y2 = rng.normal(coef['mean'], 0, ncoef)
    y2[0] -= 3e4 # we slightly reduce the mean because over multiple runs the average was a little too high.

    capfac_wind = idct(y2, n=npts)
    capfac_wind[np.where(capfac_wind < 0)] = 0
    capfac_wind[np.where(capfac_wind > 1)] = 1
    wind = capfac_wind * caps['wind']

    export_ratio = 0.3 # Ratio of potential exports to Estonian demand, for 2018 Estonia exported an amount equal to about 26% of its own demand @nordpoolMarketData

    # calculate needed capacity from dispatchable sources
    caps_dispatch = caps['oil shale'] + caps['shale gas'] + caps['nuclear']
    if caps_dispatch > 0:
        dispatch_need = smoothed_demand - wind - solar # start by calculating net load
        dispatch_need[np.where(dispatch_need < caps_dispatch)] *= (1 + export_ratio - 0.05) # include potential exports
        y4 = dct(dispatch_need)
        dispatch_need = idct(y4[:2000], n=npts)

    # nuclear
    if caps['nuclear'] > 0:
        if not variable or variable == 'capfac_nuclear_avg':
            capfac_nuclear_avg = rng.normal(0.935, 0.03, 1) # @eiaElectricPowerMonthly
        else:
            capfac_nuclear_avg = rng.normal(0.935, 0, 1)

        if capfac_nuclear_avg > 1: capfac_nuclear_avg = 1
        min_cap_nuclear = 0.2 # nuclear power plants can also be operated flexibly, @morilhatNuclearPowerPlant2019
        max_cap_nuclear = capfac_nuclear_avg

        nuclear = caps['nuclear'] / caps_dispatch * dispatch_need
        nuclear[np.where(nuclear < min_cap_nuclear * caps['nuclear'])] = min_cap_nuclear * caps['nuclear']
        nuclear[np.where(nuclear > max_cap_nuclear * caps['nuclear'])] = max_cap_nuclear * caps['nuclear']
    else:
        nuclear = 0

    # oil shale with CCS
    if caps['oil shale'] > 0:
        if not variable or variable == 'capfac_os_avg':
            capfac_os_avg = rng.normal(0.82, 0.03, 1) # @jamesCostPerformanceBaseline2019
        else:
            capfac_os_avg = rng.normal(0.82, 0, 1) # @jamesCostPerformanceBaseline2019

        if capfac_os_avg > 1: capfac_os_avg = 1
        min_cap_os = 0.4 # @irenaFlexibilityConventionalPower2019
        max_cap_os = capfac_os_avg

        os = caps['oil shale'] / caps_dispatch * dispatch_need
        os[np.where(os < min_cap_os * caps['oil shale'])] = min_cap_os * caps['oil shale']
        os[np.where(os > max_cap_os * caps['oil shale'])] = max_cap_os * caps['oil shale']
    else:
        os = 0

    # shale gas with CCS
    if caps['shale gas'] > 0:
        if not variable or variable == 'capfac_gas_avg':
            capfac_gas_avg = rng.normal(0.85, 0.03, 1) # @jamesCostPerformanceBaseline2019
        else:
            capfac_gas_avg = rng.normal(0.85, 0, 1) # @jamesCostPerformanceBaseline2019

        if capfac_gas_avg > 1: capfac_gas_avg = 1
        min_cap_gas = 0.4
        max_cap_gas = capfac_gas_avg

        shale_gas = caps['shale gas'] / caps_dispatch * dispatch_need
        shale_gas[np.where(shale_gas < min_cap_gas * caps['shale gas'])] = min_cap_gas * caps['shale gas']
        shale_gas[np.where(shale_gas > max_cap_gas * caps['shale gas'])] = max_cap_gas * caps['shale gas']
    else:
        shale_gas = 0

    produced = biomass + hydro + solar + wind + nuclear + os + shale_gas
    if not variable or variable == 'transmission_losses':
        transmission_losses = rng.normal(0.08, 0.01, 1) # @eestistatistikaametKE03ElektrienergiaBilanss
    else:
        transmission_losses = rng.normal(0.08, 0, 1) # @eestistatistikaametKE03ElektrienergiaBilanss

    produced = (1 - transmission_losses) * produced

    if plot_example:
        # plot an example of the production and demand data from a simulation
        plt.rcParams.update({'font.size': 16})
        day_start = 30
        day_end = 35
        i_start = int(day_start * SECONDS_IN_DAY / time_step)
        i_end = int(day_end * SECONDS_IN_DAY / time_step)
        fig, ax = plt.subplots(figsize=(12,9))
        plt.plot(t[i_start:i_end] / SECONDS_IN_DAY, demanded[i_start:i_end], label='Demand', color='black', linewidth=2)
        plt.plot(t[i_start:i_end] / SECONDS_IN_DAY, wind[i_start:i_end], label='Wind', linewidth=2)
        if caps['oil shale'] > 0:
            plt.plot(t[i_start:i_end] / SECONDS_IN_DAY, os[i_start:i_end], label='Oil shale', linewidth=2)
        if caps['nuclear'] > 0:
            plt.plot(t[i_start:i_end] / SECONDS_IN_DAY, nuclear[i_start:i_end], label='Nuclear', linewidth=2)
        if caps['shale gas'] > 0:
            plt.plot(t[i_start:i_end] / SECONDS_IN_DAY, shale_gas[i_start:i_end], label='Shale gas', linewidth=2)
        plt.plot(t[i_start:i_end] / SECONDS_IN_DAY, biomass[i_start:i_end] + hydro + solar[i_start:i_end], label='Other', linewidth=2)
        plt.plot(t[i_start:i_end] / SECONDS_IN_DAY, produced[i_start:i_end], label='Total production', linestyle='--')
        plt.ylabel('MW')
        plt.xlabel('Day of the year')
        plt.xlim([day_start, day_end])
        plt.legend(frameon=False, loc=1, bbox_to_anchor=(1.40, 1.0))
        plt.subplots_adjust(right=0.75)
        plt.savefig('figures/simulation_example', dpi=400)
        plt.show()

    annual_prod = {}
    annual_prod['wind'] = np.sum(wind) * time_step * (1 - transmission_losses) / 3600 # MWh
    annual_prod['biomass'] = np.sum(biomass) * time_step * (1 - transmission_losses) / 3600 # MWh
    annual_prod['hydro'] = np.sum(hydro) * time_step * (1 - transmission_losses) / 3600 # MWh
    annual_prod['solar'] = np.sum(solar) * time_step * (1 - transmission_losses) / 3600 # MWh
    annual_prod['nuclear'] = np.sum(nuclear) * time_step * (1 - transmission_losses) / 3600 # MWh
    annual_prod['oil shale'] = np.sum(os) * time_step * (1 - transmission_losses) / 3600 # MWh
    annual_prod['shale gas'] = np.sum(shale_gas) * time_step * (1 - transmission_losses) / 3600 # MWh
    annual_prod['total'] = np.sum(produced) * time_step * (1 - transmission_losses) / 3600 # MWh

    return produced, annual_prod


def calc_price(t, rng, variable=None):
    npts = t.shape[0]
    SECONDS_IN_YR = 31536000
    SECONDS_IN_DAY = 86400
    time_step = SECONDS_IN_YR / npts

    coef = pd.read_csv('data/price_dct_coefficients.csv', delimiter=';') # loading the parameters for the coefficients

    ncoef = coef.shape[0]
    if not variable or variable == 'price':
        y = rng.normal(coef['mean'], coef['stdev'], ncoef)
    else:
        y = rng.normal(coef['mean'], 0, ncoef)

    price = idct(y, n=npts)
    return price


def demand(t, rng, variable=None):
    npts = t.shape[0]
    SECONDS_IN_YR = 31536000
    SECONDS_IN_DAY = 86400
    time_step = SECONDS_IN_YR / npts
    ncoef = 5000 # we found that only the first 5000 or so coefficients actually had a meaningful impact

    # loading the parameters for the coefficients
    coef = pd.read_csv('data/demand_dct_coefficients.csv', delimiter=';')
    coef = coef.head(ncoef)

    if not variable or variable == 'demand':
        y = rng.normal(coef['mean'], coef['stdev'], ncoef)
    else:
        y = rng.normal(coef['mean'], 0, ncoef)

    demanded = idct(y, n=npts)
    return demanded


def export(surplus, demanded):
    export_ratio = 0.3 # Ratio of potential exports to Estonian demand, for 2018 Estonia exported an amount equal to about 26% of its own demand @nordpoolMarketData
    exported = np.copy(surplus)
    exported[np.where(exported < 0)] = 0
    surplus_too_high = np.where(exported > demanded * export_ratio)
    exported[surplus_too_high] = demanded[surplus_too_high] * export_ratio
    return exported


def calc_cost(caps, prod, fraction_lost, storage_tech, rng, price, to_storage, variable=None):
    SECONDS_IN_YR = 31536000
    time_step = SECONDS_IN_YR / price.shape[0]
    npts = 1
    cost = np.zeros((npts,))

    # wind
    if caps['wind'] > 0:
        if caps['wind'] >= 400:
            capacity_wind_on = 400 # assuming 400 MW of the wind capacity is onshore
        else:
            capacity_wind_on = caps['wind']
        caps_wind_on = caps['wind'] * capacity_wind_on / caps['wind']
        prod_wind_on = prod['wind'] * capacity_wind_on / prod['wind']
        cost_wind_on = co.cost_wind_on(npts, caps_wind_on, prod_wind_on, fraction_lost, rng, variable=variable)
        cost += cost_wind_on * prod['wind'] * capacity_wind_on / prod['wind'] / prod['total'] # we use a weighted average to combine the levelized costs of each system component
        caps_wind_off = caps['wind'] * (1 - capacity_wind_on / caps['wind'])
        prod_wind_off = prod['wind'] * (1 - capacity_wind_on / prod['wind'])
        cost_wind_off = co.cost_wind_off(npts, caps_wind_off, prod_wind_off, fraction_lost, rng, variable=variable)
        cost += cost_wind_off * prod['wind'] * (1 - capacity_wind_on / prod['wind']) / prod['total']

    # solar
    cost_solar = co.cost_solar(npts, caps['solar'], prod['solar'], rng, variable=variable) * prod['solar'] / prod['total']
    cost += cost_solar

    # nuclear
    cost_nuclear = co.cost_nuclear(npts, caps['nuclear'], prod['nuclear'], rng, variable=variable) * prod['nuclear'] / prod['total']
    cost += cost_nuclear

    # oil shale
    cost_os_noccs, cost_os_ccs_avg, cost_os_ccs_next = co.cost_os(npts, caps['oil shale'], prod['oil shale'], rng, variable=variable) * prod['oil shale'] / prod['total']
    cost += cost_os_ccs_avg

    # shale gas
    cost_gas_noccs, cost_gas_ccs_avg, cost_gas_ccs_next = co.cost_os_gas(npts, caps['shale gas'], prod['shale gas'], rng, variable=variable) * prod['shale gas'] / prod['total']
    cost += cost_gas_ccs_avg

    # biomass
    if not variable or variable == 'cost_biomass':
        cost_biomass = rng.normal(100, 25, npts) * prod['biomass'] / prod['total'] # @irenaBiomassPowerGeneration2012
    else:
        cost_biomass = rng.normal(100, 0, npts) * prod['biomass'] / prod['total'] # @irenaBiomassPowerGeneration2012
    cost += cost_biomass

    # hydro
    if not variable:
        cost_hydro = rng.normal(80, 20, npts) * prod['hydro'] / prod['total'] # @Hydropower
    else:
        cost_hydro = rng.normal(80, 0, npts) * prod['hydro'] / prod['total'] # @Hydropower
    cost += cost_hydro

    # storage
    if storage_tech == 'uphes':
        cost_storage, cost_storage_noarb = co.cost_storage_uphes(npts, caps['storage_energy'], caps['storage_power'], prod['total'], rng, price, to_storage, variable=variable)
    elif storage_tech == 'hydrogen':
        cost_storage = co.cost_storage_hydrogen(npts, caps['storage_energy'], caps['storage_power'], prod['total'], rng, variable=variable)
    else:
        throw(SimulationError('{} is not currently a valid storage technology. Add a function for calculating its cost.'))
    cost += cost_storage
    dif_from_arbitrage = cost_storage - cost_storage_noarb

    return cost, dif_from_arbitrage


def monte_carlo(npts, caps, processes=3, variable=None):
    results = []
    with multiprocessing.Pool(processes) as pool:
        raw_results = [pool.apply_async(simulation, args=(caps, variable)) for i in range(npts)]
        for r in raw_results:
            try:
                results.append(r.get(timeout=90))
            except multiprocessing.TimeoutError:
                pass

    results = np.asarray(results)
    total_production = results[:,0]
    total_excess = results[:,1]
    total_shortage = results[:,2]
    total_unused = results[:,3]
    net_surplus = results[:,4]
    cost = results[:,5]
    dif_from_arbitrage = results[:,6]

    return total_production, total_excess, total_shortage, total_unused, net_surplus, cost, dif_from_arbitrage


def monte_carlo_surplus_duration(npts, istart, caps, processes=3):
    run = []
    probs = []
    surplus = []
    with multiprocessing.Pool(processes) as pool:
        raw_results = [pool.apply_async(surplus_duration, args=(caps, i)) for i in range(istart, istart + npts)]
        for r in raw_results:
            try:
                result = r.get(timeout=90)
                run.append(result[0])
                probs.append(result[1])
                surplus.append(result[2])
            except multiprocessing.TimeoutError:
                pass

    run = np.vstack(run).flatten()
    probs = np.vstack(probs).flatten()
    surplus = np.vstack(surplus).flatten()
    return run, probs, surplus


def monte_carlo_demand_duration(npts, istart, processes=3):
    run = []
    probs = []
    demand = []
    with multiprocessing.Pool(processes) as pool:
        raw_results = [pool.apply_async(demand_duration, args=(i,)) for i in range(istart, istart + npts)]
        for r in raw_results:
            try:
                result = r.get(timeout=90)
                run.append(result[0])
                probs.append(result[1])
                demand.append(result[2])
            except multiprocessing.TimeoutError:
                pass

    run = np.vstack(run).flatten()
    probs = np.vstack(probs).flatten()
    demand = np.vstack(demand).flatten()
    return run, probs, demand


def find_demand_coefficients():
    # perform Discrete Cosine Transform on actual demand data from the past 5 years and calculate the mean and standard deviation of each DCT coefficient.
    SECONDS_IN_YR = 31536000
    time_step = 10 # seconds
    t = np.arange(0, SECONDS_IN_YR / time_step) * time_step # current second in the year
    years = ['2016', '2017', '2018', '2019', '2020']

    y = np.zeros((t.shape[0], 5))
    for i, yr in enumerate(years):
        comp = pd.read_csv('data/electricity-production and consumption_{}.csv'.format(yr), delimiter=';') # actual data from @eleringasEleringLive
        comp['second'] = 0
        comp.loc[:, 'second'] = (comp['Ajatempel (UTC)'] - comp['Ajatempel (UTC)'].iloc[0])
        spline = interpolate.splrep(comp['second'], comp['Tarbimine'], s=0)
        demand_actual = interpolate.splev(t, spline)
        y[:,i] = dct(demand_actual)

    res = np.hstack((np.mean(y, axis=1).reshape(-1,1), np.std(y, axis=1).reshape(-1,1)))
    df = pd.DataFrame(data=res, columns=['mean', 'stdev'])
    df.to_csv('data/demand_dct_coefficients.csv', sep=';')


def find_wind_coefficients():
    # perform Discrete Cosine Transform on actual wind production data from the past 5 years and calculate the mean and standard deviation of each DCT coefficient.
    SECONDS_IN_YR = 31536000
    time_step = 10 # seconds
    t = np.arange(0, SECONDS_IN_YR / time_step) * time_step # current second in the year
    years = ['2016', '2017', '2018', '2019', '2020']
    wind_capacity = 311 # MW (capacity between 2016 and 2019) @eestistatistikaametKE032ElektrijaamadeVoimsus

    y = np.zeros((t.shape[0], 5))
    for i, yr in enumerate(years):
        comp = pd.read_csv('data/electricity-production-wind parks_{}.csv'.format(yr), delimiter=';') # actual data from @eleringasEleringLive
        comp['second'] = 0
        comp.loc[:, 'second'] = (comp['Ajatempel (UTC)'] - comp['Ajatempel (UTC)'].iloc[0])
        spline = interpolate.splrep(comp['second'], comp['Tuuleparkide toodang'], s=0)
        capfac_actual = interpolate.splev(t, spline) / wind_capacity
        y[:,i] = dct(capfac_actual)

    res = np.hstack((np.mean(y, axis=1).reshape(-1,1), np.std(y, axis=1).reshape(-1,1)))
    df = pd.DataFrame(data=res, columns=['mean', 'stdev'])
    df.to_csv('data/wind_dct_coefficients.csv', sep=';')


def find_price_coefficients():
    # perform Discrete Cosine Transform on actual Nordpool price data for Estonia from the past 5 years and calculate the mean and standard deviation of each DCT coefficient.
    SECONDS_IN_YR = 31536000
    time_step = 10 # seconds
    t = np.arange(0, SECONDS_IN_YR / time_step) * time_step # current second in the year
    years = ['2017', '2018', '2019', '2020']

    y = np.zeros((t.shape[0], 5))
    for i, yr in enumerate(years):
        comp = pd.read_csv('electricity-production and consumption_{}.csv'.format(yr), delimiter=';') # actual data from @eleringasEleringLive
        comp['second'] = 0
        comp.loc[:, 'second'] = (comp['Ajatempel (UTC)'] - comp['Ajatempel (UTC)'].iloc[0])
        spline = interpolate.splrep(comp['second'], comp['NPS Eesti'], s=0)
        price_actual = interpolate.splev(t, spline)
        y[:,i] = dct(price_actual)

    res = np.hstack((np.mean(y, axis=1).reshape(-1,1), np.std(y, axis=1).reshape(-1,1)))
    df = pd.DataFrame(data=res, columns=['mean', 'stdev'])
    df.to_csv('price_dct_coefficients.csv', sep=';')


def single_sim(show_plot=False, plot_example=False):
    caps = {
        "name": "Renewables with oil shale",
        "storage_power": 0,
        "storage_energy": 0,
        "wind": 1900,
        "solar": 40,
        "nuclear": 0,
        "oil shale": 415,
        "shale gas": 180
    }

    result = simulation(caps, show_plot=show_plot, plot_example=plot_example)
    print(result)


def storage_effect(run_simulation=True):
    caps_grid = np.logspace(2, 6, 20)
    caps_grid = np.insert(caps_grid, 0, 0.)
    powers_grid = np.logspace(2, 4, 10)
    storage_caps, storage_powers = np.meshgrid(caps_grid, powers_grid)
    storage_caps = storage_caps.flatten()
    storage_powers = storage_powers.flatten()
    wind_caps = 3800

    outfile = 'data/storage_effect.csv'

    if run_simulation:
        if not os.path.exists(outfile):
            with open(outfile, 'a+') as f:
                f.seek(0)
                f.truncate()
                f.write('Storage energy;Storage power;Wind capacity;Total production;Total surplus;Total shortage;Total unused;Net surplus;Average cost\n')

        for j, st in enumerate(storage_caps):
            print('Storage: {:.0f} MWh'.format(st))
            caps = {
                "storage_power": storage_powers[j],
                "storage_energy": st,
                "wind": wind_caps,
                "solar": 40,
                "nuclear": 0,
                "oil shale": 0
            }
            total_production, total_excess, total_shortage, total_unused, net_surplus, cost, dif_from_arbitrage = monte_carlo(200, caps)

            with open(outfile, 'a+') as f:
                for i in range(total_production.shape[0]):
                    f.write(';'.join([str(st), str(storage_powers[j]), str(wind_caps), str(total_production[i]), str(total_excess[i]), str(total_shortage[i]), str(total_unused[i]), str(net_surplus[i]), str(cost[i])]) + '\n')

    df = pd.read_csv(outfile, delimiter=';')
    df.loc[df['Storage energy'] == 0, 'Storage energy'] = 1
    grp = df.groupby(['Storage energy', 'Storage power']).mean()
    grp.reset_index(inplace=True)
    grp.to_csv('data/storage_averages.csv', sep=';')

    fig, axs = plt.subplots(3, 1, figsize=(9,14))

    clrs1 = axs[0].scatter(grp['Storage energy'], -grp['Net surplus']/1000, c=grp['Storage power'], cmap=plt.get_cmap('copper'), norm=matplotlib.colors.LogNorm())
    clrs2 = axs[1].scatter(grp['Storage energy'], -grp['Total shortage']/1000, c=grp['Storage power'], cmap=plt.get_cmap('copper'), norm=matplotlib.colors.LogNorm())
    clrs3 = axs[2].scatter(grp['Storage energy'], grp['Total unused']/1000, c=grp['Storage power'], cmap=plt.get_cmap('copper'), norm=matplotlib.colors.LogNorm())
    cbar1 = fig.colorbar(clrs1, ax=axs[0])
    cbar2 = fig.colorbar(clrs2, ax=axs[1])
    cbar3 = fig.colorbar(clrs3, ax=axs[2])
    cbar1.set_label('Storage power (MW)')
    cbar2.set_label('Storage power (MW)')
    cbar3.set_label('Storage power (MW)')

    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    axs[2].set_xscale('log')

    axs[0].set_ylabel('Net deficit (GWh)')
    axs[1].set_ylabel('Total imported (GWh)')
    axs[2].set_ylabel('Total overproduction (GWh)')

    axs[-1].set_xlabel('Storage capacity (MWh)')
    plt.subplots_adjust(left=0.17)
    plt.savefig('figures/renewable_storage_ratios', dpi=400)
    plt.show()


def compare_systems(run_simulation=True):
    with open('systems.json', 'r') as file:
        systems = json.load(file) # all capacities in MW, except for storage_energy which has units of MWh

    outfile = 'data/compare_systems.csv'
    nruns = 3003

    if run_simulation:
        if not os.path.exists(outfile):
            with open(outfile, 'a+') as f:
                f.seek(0)
                f.truncate()
                f.write('System;Total production;Total surplus;Total shortage;Total unused;Net surplus;Average cost;Difference from arbitrage\n')

        for i, caps in enumerate(systems):
            print('Performing simulation for {}'.format(caps['name']))
            total_production, total_excess, total_shortage, total_unused, net_surplus, cost, dif_from_arbitrage = monte_carlo(nruns, caps)

            with open(outfile, 'a+') as f:
                for i in range(total_production.shape[0]):
                    f.write(';'.join([caps['name'], str(total_production[i]), str(total_excess[i]), str(total_shortage[i]), str(total_unused[i]), str(net_surplus[i]), str(cost[i]), str(dif_from_arbitrage[i])]) + '\n')

    # creating plot
    df = pd.read_csv(outfile, delimiter=';')

    fig, axs = plt.subplots(4, 1, figsize=(9,18))

    colors = ['#7dc888', '#489b48', '#267b26', '#005e0d', '#b50d0d', '#4242e3', '#6a2525', '#dc961b', '#a812cd', '#15d4c9']
    labels = []
    for i, caps in enumerate(systems):
        labels.append(caps['name'])

        bp1 = axs[0].boxplot(df.loc[df.System == caps['name'], 'Average cost'], positions=[i], whis=[5,95], widths=0.2, patch_artist=True, showfliers=False)
        bp2 = axs[1].boxplot(df.loc[df.System == caps['name'], 'Net surplus'] / 1000, positions=[i], whis=[5,95], widths=0.2, patch_artist=True, showfliers=False)
        bp3 = axs[2].boxplot(df.loc[df.System == caps['name'], 'Total shortage'] / -1000, positions=[i], whis=[5,95], widths=0.2, patch_artist=True, showfliers=False)
        bp4 = axs[3].boxplot(df.loc[df.System == caps['name'], 'Total unused'] / 1000, positions=[i], whis=[5,95], widths=0.2, patch_artist=True, showfliers=False)

        for element in ['boxes', 'whiskers', 'fliers', 'means', 'caps']:
            plt.setp(bp1[element], color=colors[i])
            plt.setp(bp2[element], color=colors[i])
            plt.setp(bp3[element], color=colors[i])
            plt.setp(bp4[element], color=colors[i])

        for patch in bp1['boxes']:
            patch.set(facecolor=colors[i])
        for patch in bp2['boxes']:
            patch.set(facecolor=colors[i])
        for patch in bp3['boxes']:
            patch.set(facecolor=colors[i])
        for patch in bp4['boxes']:
            patch.set(facecolor=colors[i])

        plt.setp(bp1['medians'], color='white')
        plt.setp(bp2['medians'], color='white')
        plt.setp(bp3['medians'], color='white')
        plt.setp(bp4['medians'], color='white')
        plt.setp(bp1['whiskers'], linewidth=2.0)
        plt.setp(bp2['whiskers'], linewidth=2.0)
        plt.setp(bp3['whiskers'], linewidth=2.0)
        plt.setp(bp4['whiskers'], linewidth=2.0)

    labels = ['\n'.join(wrap(l, 20)) for l in labels]

    axs[0].set_ylabel('Cost (EUR / MWh)')
    axs[1].set_ylabel('Net surplus (GWh)')
    axs[2].set_ylabel('Total import (GWh)')
    axs[3].set_ylabel('Total overproduction (GWh)')
    axs[0].set_xticklabels([])
    axs[1].set_xticklabels([])
    axs[2].set_xticklabels([])
    axs[-1].set_xticklabels(labels, rotation=90)
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig('figures/sim_results', dpi=400)
    plt.show()

def calc_average_probability(df, name):
    df1 = df.loc[df.System == name, :].copy(deep=True)
    df1.drop(columns='System', inplace=True)
    df1 = df1.groupby(by='Probability', as_index=False).quantile(0.5)
    return df1['Probability'], df1['Surplus']

def surplus_duration_curve(run_simulation=True):
    HRS_PER_YEAR = 8760 # h/yr

    with open('systems.json', 'r') as file:
        systems = json.load(file) # all capacities in MW, except for storage_energy which has units of MWh

    outfile = 'data/surplus_duration_curve.csv'
    nruns = 2900

    if run_simulation:
        istart = 0
        if not os.path.exists(outfile):
            with open(outfile, 'a+') as f:
                f.seek(0)
                f.truncate()
                f.write('System;Run;Probability;Surplus\n')
        else:
            tmp = pd.read_csv(outfile, delimiter=';')
            istart = int(tmp['Run'].max()) + 1

        for i, caps in enumerate(systems):
            print('Performing simulation for {}'.format(caps['name']))
            run, probs, surplus = monte_carlo_surplus_duration(nruns, istart, caps)

            with open(outfile, 'a+') as f:
                for i in range(run.shape[0]):
                    f.write(';'.join([caps['name'], str(run[i]), str(probs[i]), str(surplus[i])]) + '\n')

    # creating plot
    df = pd.read_csv(outfile, delimiter=';')

    fig, axs = plt.subplots(figsize=(10,6))

    colors = ['#7dc888', '#489b48', '#267b26', '#005e0d', '#b50d0d', '#4242e3', '#6a2525', '#dc961b', '#a812cd', '#15d4c9']
    for i, caps in enumerate(systems):
        probability, surplus = calc_average_probability(df, caps['name'])
        surplus = np.flip(surplus)
        if caps['name'] == "Nuclear":
            axs.plot(probability / 100 * HRS_PER_YEAR, surplus, color=colors[i], linestyle='dashed', label='\n'.join(wrap(caps['name'], 20)))
        else:
            axs.plot(probability / 100 * HRS_PER_YEAR, surplus, color=colors[i], label='\n'.join(wrap(caps['name'], 20)))

    axs.set_xlabel('Hours of the year')
    axs.set_ylabel('Surplus/deficit (MW)')
    axs.set_xlim(0, HRS_PER_YEAR)
    axs.legend(frameon=False, bbox_to_anchor=(1.6, 0.5), loc='center right')
    plt.subplots_adjust(right=0.65)
    plt.savefig('figures/surplus_duration_curve', dpi=400)
    plt.show()


def demand_duration_curve(run_simulation=True):
    HRS_PER_YEAR = 8760 # h/yr

    outfile = 'data/demand_duration_curve.csv'
    nruns = 3

    if run_simulation:
        istart = 0
        if not os.path.exists(outfile):
            with open(outfile, 'a+') as f:
                f.seek(0)
                f.truncate()
                f.write('Run;Probability;Demand\n')
        else:
            tmp = pd.read_csv(outfile, delimiter=';')
            istart = int(tmp['Run'].max()) + 1

        print('Generating demand curves')
        run, probs, demand = monte_carlo_demand_duration(nruns, istart)

        with open(outfile, 'a+') as f:
            for i in range(run.shape[0]):
                f.write(';'.join([str(run[i]), str(probs[i]), str(demand[i])]) + '\n')

    # creating plot
    df = pd.read_csv(outfile, delimiter=';')

    fig, axs = plt.subplots()

    runs = df['Run'].unique().tolist()
    for r in runs:
        demand = np.flip(df.loc[df.Run == r, 'Demand'])
        axs.plot(df.loc[df.Run == r, 'Probability'] / 100 * HRS_PER_YEAR, demand, color='grey', alpha=0.5)

    years = ['2020', '2019', '2018', '2017', '2016']
    for yr in years:
        comp = pd.read_csv('data/electricity-production and consumption_{}.csv'.format(yr), delimiter=';')
        probability, demand = probability_dist(comp['Tarbimine'])
        demand = np.flip(demand)
        axs.plot(probability / 100 * HRS_PER_YEAR, demand, linewidth=2, label='Actual data ({})'.format(yr))

    axs.plot([], [], color="grey", label='Generated data') # Add label for generated data

    axs.set_xlabel('Hours of the year')
    axs.set_ylabel('Demand (MW)')
    axs.set_xlim(0, HRS_PER_YEAR)
    axs.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('figures/demand_duration_curve', dpi=400)
    plt.show()


def wind_penetration_cost(run_simulation=True):
    wind_caps = np.linspace(0, 3080, 14)
    nuclear_caps = np.linspace(1090, 200, 14)
    oil_shale_caps = np.linspace(1233, 227, 14)
    shale_gas_caps = 180 # MW_electric
    nruns = 2002

    outfile = 'data/wind_penetration.csv'

    if run_simulation:
        if not os.path.exists(outfile):
            with open(outfile, 'a+') as f:
                f.seek(0)
                f.truncate()
                f.write('Wind capacity;Nuclear capacity;Oil shale capacity;Total production;Total surplus;Total shortage;Total unused;Net surplus;Average cost;Difference from arbitrage\n')

        for i, wind in enumerate(wind_caps):
            print('Wind: {:.0f} MW'.format(wind))
            caps = {
                "storage_power": 0,
                "storage_energy": 0,
                "wind": wind,
                "solar": 40,
                "nuclear": nuclear_caps[i],
                "oil shale": 0,
                "shale gas": 0
            }
            total_production, total_excess, total_shortage, total_unused, net_surplus, cost, dif_from_arbitrage = monte_carlo(nruns, caps)

            with open(outfile, 'a+') as f:
                for j in range(total_production.shape[0]):
                    f.write(';'.join([str(wind), str(nuclear_caps[i]), '0', str(total_production[j]), str(total_excess[j]), str(total_shortage[j]), str(total_unused[j]), str(net_surplus[j]), str(cost[j]), str(dif_from_arbitrage[j])]) + '\n')

            caps = {
                "storage_power": 0,
                "storage_energy": 0,
                "wind": wind,
                "solar": 40,
                "nuclear": 0,
                "oil shale": oil_shale_caps[i] - shale_gas_caps,
                "shale gas": shale_gas_caps
            }
            total_production, total_excess, total_shortage, total_unused, net_surplus, cost, dif_from_arbitrage = monte_carlo(nruns, caps)

            with open(outfile, 'a+') as f:
                for j in range(total_production.shape[0]):
                    f.write(';'.join([str(wind), '0', str(oil_shale_caps[i]), str(total_production[j]), str(total_excess[j]), str(total_shortage[j]), str(total_unused[j]), str(net_surplus[j]), str(cost[j]), str(dif_from_arbitrage[j])]) + '\n')

    # creating plot
    def cap_to_percent(x):
        wind_capfac = 0.2679
        solar = 0.11 * 40 # 40 MW of capacity multiplied by the capacity factor
        biomass_and_waste = 160 # MW
        remaining = 3080 * wind_capfac
        total_cap = biomass_and_waste + solar + remaining
        return x * wind_capfac / total_cap * 100

    def percent_to_cap(x):
        wind_capfac = 0.2679
        solar = 0.11 * 40 # 40 MW of capacity multiplied by the capacity factor
        biomass_and_waste = 160 # MW
        remaining = 3080 * wind_capfac
        total_cap = biomass_and_waste + solar + remaining
        return x * wind_capfac / 100 * total_cap

    df = pd.read_csv(outfile, delimiter=';')
    grp = df.groupby(['Wind capacity', 'Nuclear capacity', 'Oil shale capacity']).mean()
    grp.reset_index(inplace=True)
    grp.to_csv('data/wind_penetration_averages.csv', sep=';')

    fig, axs = plt.subplots(2, 1, figsize=(7,11))
    secax_top = axs[0].secondary_xaxis('top', functions=(cap_to_percent, percent_to_cap))
    secax_bottom = axs[1].secondary_xaxis('top', functions=(cap_to_percent, percent_to_cap))

    axs[0].scatter(grp.loc[grp['Oil shale capacity'] == 0, 'Wind capacity'], grp.loc[grp['Oil shale capacity'] == 0, 'Average cost'], label='Nuclear')
    axs[0].scatter(grp.loc[grp['Nuclear capacity'] == 0, 'Wind capacity'], grp.loc[grp['Nuclear capacity'] == 0, 'Average cost'], label='Oil shale and pyrolysis gas')
    axs[1].scatter(grp.loc[grp['Oil shale capacity'] == 0, 'Wind capacity'], grp.loc[grp['Oil shale capacity'] == 0, 'Net surplus'] / 1000, label='Nuclear')
    axs[1].scatter(grp.loc[grp['Nuclear capacity'] == 0, 'Wind capacity'], grp.loc[grp['Nuclear capacity'] == 0, 'Net surplus'] / 1000, label='Oil shale and pyrolysis gas')

    axs[1].set_xlabel('Wind capacity (MW)')
    axs[0].set_ylabel('Cost (EUR/MWh)')
    axs[1].set_ylabel('Net surplus (GWh)')
    secax_top.set_xlabel('Wind percentage of total generation (%)')

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('figures/wind_penetration', dpi=400)
    plt.show()


def sensitivity_analysis(run_simulation=True):
    variables_split = {
        'Renewables - 100 GWh storage': ['capital_solar_ee', 'variable_solar', 'grid_investments', 'wind_on',
            'wind_off', 'capital_power', 'capital_energy', 'variable_storage', 'cost_biomass', 'demand',
            'price', 'transmission_losses', 'capfac_wind', 'capfac_solar', 'biomass', 'construct_time uphes'],
        'Renewables with oil shale and pyrolysis gas': ['co2_os', 'os o_and_m', 'co2_credits', 'construction_os',
            'trans_storage', 'ccs_avg', 'co2_gas', 'gas o_and_m', 'construction_gas', 'capital_solar_ee',
            'variable_solar', 'grid_investments', 'wind_on', 'wind_off', 'cost_biomass', 'demand', 'price',
            'transmission_losses', 'capfac_gas_avg', 'capfac_os_avg', 'capfac_wind', 'capfac_solar', 'biomass',
            'construct_time os', 'construct_time gas'],
        'Renewables, nuclear, and pyrolysis gas': ['co2_credits', 'trans_storage', 'ccs_avg', 'co2_gas',
            'gas o_and_m', 'construction_gas', 'capital_solar_ee', 'variable_solar', 'grid_investments',
            'wind_on', 'wind_off', 'cost_biomass', 'demand', 'price', 'transmission_losses',
            'capfac_gas_avg', 'capfac_wind', 'capfac_solar', 'biomass', 'capital_smr', 'fuel nuclear',
            'o_and_m nuclear', 'capfac_nuclear_avg', 'construct_time nuclear']
    }

    variables_all = ['co2_os', 'os o_and_m', 'co2_credits', 'construction_os', 'trans_storage',
        'ccs_avg', 'co2_gas', 'gas o_and_m', 'construction_gas', 'capital_solar_ee',
        'variable_solar', 'grid_investments', 'wind_on', 'wind_off', 'capital_power',
        'capital_energy', 'variable_storage', 'cost_biomass', 'demand', 'price', 'transmission_losses',
        'capfac_gas_avg', 'capfac_os_avg', 'capfac_wind', 'capfac_solar', 'biomass', 'capital_smr',
        'fuel nuclear', 'o_and_m nuclear', 'capfac_nuclear_avg']

    variables_plot = ['co2_os', 'os o_and_m', 'co2_credits', 'construction_os', 'ccs_avg',
        'trans_storage', 'co2_gas', 'gas o_and_m', 'construction_gas', 'capital_smr', 'fuel nuclear',
        'o_and_m nuclear', 'wind_off', 'capital_power', 'capital_energy', 'variable_storage',
        'cost_biomass', 'demand', 'price', 'grid_investments', 'transmission_losses',
        'capfac_gas_avg', 'capfac_os_avg', 'capfac_wind', 'capfac_nuclear_avg', 'biomass']

    labels_dict = {
        'co2_os': 'OS CO2 emission factor',
        'os o_and_m': 'OS operating costs',
        'co2_credits': 'CO2 credit price',
        'construction_os': 'OS construction cost',
        'trans_storage': 'CO2 transport and storage',
        'ccs_avg': 'CCS cost',
        'co2_gas': 'Gas CO2 emission factor',
        'gas o_and_m': 'Gas operating costs',
        'construction_gas': 'Gas construction cost',
        'grid_investments': 'Grid investments',
        'wind_off': 'Wind LCOE',
        'capital_power': 'Storage power cost',
        'capital_energy': 'Storage cost per energy',
        'variable_storage': 'Storage operating costs',
        'cost_biomass': 'Biomass LCOE',
        'demand': 'Electricity demand',
        'price': 'Electricity price',
        'transmission_losses': 'Transmission losses',
        'capfac_gas_avg': 'Gas capacity factor',
        'capfac_os_avg': 'OS capacity factor',
        'capfac_wind': 'Wind capacity factor',
        'biomass': 'Biomass capacity',
        'capital_smr': 'Nuclear construction cost',
        'fuel nuclear': 'Nuclear fuel cost',
        'o_and_m nuclear': 'Nuclear operating costs',
        'capfac_nuclear_avg': 'Nuclear capacity factor'
    }
    # labels = ['\n'.join(wrap(labels_dict[v], 20)) for v in variables_plot]
    labels = [labels_dict[v] for v in variables_plot]

    with open('systems.json', 'r') as file:
        systems = json.load(file) # all capacities in MW, except for storage_energy which has units of MWh

    fig, axs = plt.subplots(2, 1, figsize=(11,10))

    systems_to_run = ['Renewables - 100 GWh storage', 'Renewables with oil shale and pyrolysis gas', 'Renewables, nuclear, and pyrolysis gas']
    colors = ['#005e0d', '#dc961b', '#15d4c9']
    bar_width = 0.8 / len(systems_to_run)
    legend_handles = []
    for isys, system_name in enumerate(systems_to_run):
        caps = {}
        for s in systems:
            if s['name'] == system_name:
                caps = s

        outfile = 'data/sensitivity_analysis.csv'
        nruns = 100

        print('\n{}'.format(caps['name']))

        if run_simulation:
            if not os.path.exists(outfile):
                with open(outfile, 'a+') as f:
                    f.seek(0)
                    f.truncate()
                    f.write('System;Variable name;Total production;Total surplus;Total shortage;Total unused;Net surplus;Average cost;Difference from arbitrage\n')

            for var in variables_split[system_name]:
                print('    {}'.format(var))
                total_production, total_excess, total_shortage, total_unused, net_surplus, cost, dif_from_arbitrage = monte_carlo(nruns, caps, variable=var)

                with open(outfile, 'a+') as f:
                    for i in range(total_production.shape[0]):
                        f.write(';'.join([caps['name'], var, str(total_production[i]), str(total_excess[i]), str(total_shortage[i]), str(total_unused[i]), str(net_surplus[i]), str(cost[i]), str(dif_from_arbitrage[i])]) + '\n')

        # creating plot
        df = pd.read_csv(outfile, delimiter=';')

        x = np.arange(len(variables_plot)) - 0.4 + bar_width * isys + 0.5 * bar_width

        metrics = ['Average cost', 'Net surplus']
        for i, metric in enumerate(metrics):
            for j, var in enumerate(variables_plot):
                metric_median = 0
                metric_median = df.loc[(df['System'] == system_name) & (df['Variable name'] == var), metric].median()
                if (df.loc[(df['System'] == system_name) & (df['Variable name'] == var), metric] - metric_median).max() > 1e-5:
                    if metric == 'Average cost':
                        bp = axs[i].boxplot(df.loc[(df['System'] == system_name) & (df['Variable name'] == var), metric] - metric_median, positions=[x[j]], whis=[5,95], widths=0.8*bar_width, patch_artist=True, showfliers=False)
                    else:
                        bp = axs[i].boxplot((df.loc[(df['System'] == system_name) & (df['Variable name'] == var), metric] - metric_median) / 1000, positions=[x[j]], whis=[5,95], widths=0.8*bar_width, patch_artist=True, showfliers=False)

                    for element in ['boxes', 'whiskers', 'fliers', 'means', 'caps']:
                        plt.setp(bp[element], color=colors[isys])

                    for patch in bp['boxes']:
                        patch.set(facecolor=colors[isys])

                    plt.setp(bp['medians'], color='white')
                    plt.setp(bp['whiskers'], linewidth=2.0)

        legend_handles.append(matplotlib.patches.Patch(color=colors[isys], label='\n'.join(wrap(systems_to_run[isys], 20))))

    x_ticks = np.arange(len(variables_plot))

    axs[0].plot([-2, 50], [0, 0], color='grey', linewidth=0.5, alpha=0.6)
    axs[1].plot([-2, 50], [0, 0], color='grey', linewidth=0.5, alpha=0.6)

    axs[0].set_ylabel('Cost (EUR / MWh)')
    axs[1].set_ylabel('Net surplus (GWh)')
    axs[0].set_xlim([-0.5, len(variables_plot) + 0.5])
    axs[1].set_xlim([-0.5, len(variables_plot) + 0.5])
    axs[0].set_xticks(x_ticks)
    axs[1].set_xticks(x_ticks)
    axs[0].set_xticklabels([])
    axs[-1].set_xticklabels(labels, rotation=90)
    axs[1].legend(handles=legend_handles, frameon=False, fontsize=11)
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.savefig('figures/sensitivity_analysis', dpi=400)
    plt.show()


if __name__ == '__main__':
    # find_demand_coefficients()
    # find_wind_coefficients()
    # find_price_coefficients()
    # single_sim(show_plot=False, plot_example=True)
    # storage_effect(run_simulation=False)
    # compare_systems(run_simulation=False)
    # wind_penetration_cost(run_simulation=False)
    # surplus_duration_curve(run_simulation=False)
    # demand_duration_curve(run_simulation=False)
    sensitivity_analysis(run_simulation=False)
