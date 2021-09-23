"""
Comparison of oil shale + CCS and other electricity production methods
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cost_models as cm

def plot_lcoe(npts):
    rng = np.random.default_rng()
    USD_TO_EUR = 0.876 # OECD data, 2020
    HRS_PER_YEAR = 8760 # h/yr

    years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
    price_list = []
    for y in years:
        price_list.append(pd.read_csv('electricity-production and consumption_{}.csv'.format(y), delimiter=';'))
    price_ee = pd.concat(price_list, sort=False)
    price_ee = price_ee.loc[price_ee['NPS Eesti'].notna(), :]
    price_global = pd.read_csv('electricity_prices_worldwide.csv', header=3, delimiter=';')

    os_capacity = 600
    os_capfac = rng.normal(0.82, 0.05, npts) # @jamesCostPerformanceBaseline2019
    os_production = os_capacity * os_capfac * HRS_PER_YEAR
    os_noccs, os_ccs_avg, os_ccs_best = cm.cost_os(npts, os_capacity, os_production, rng)
    os_nocredits, _, _ = cm.cost_os(npts, os_capacity, os_production, rng, credits=False)

    gas_capacity = 180
    gas_capfac = rng.normal(0.82, 0.05, npts)
    gas_production = gas_capacity * gas_capfac * HRS_PER_YEAR
    gas_noccs, gas_ccs_avg, gas_ccs_best = cm.cost_os_gas(npts, gas_capacity, gas_production, rng)

    solar_global_max = 163 * USD_TO_EUR # EUR/MWh, @irenaRenewablePowerGeneration2019 pg 13
    solar_global_min = 40 * USD_TO_EUR
    solar_global_avg = 56 * USD_TO_EUR

    solar_capacity = 40 # MW
    solar_potential = rng.normal(1, 0.04, npts) * 1000 # MWh/MW_rated/year, @theworldbankgroupGlobalSolarAtlas
    solar_production = solar_capacity * solar_potential
    solar_ee = cm.cost_solar(npts, solar_capacity, solar_production, rng, include_grid_cost=False)

    wind_on_global_max = 70 * USD_TO_EUR # EUR/MWh, @irenaRenewablePowerGeneration2019 pg 13
    wind_on_global_min = 29 * USD_TO_EUR
    wind_on_global_avg = 39 * USD_TO_EUR
    variable_wind_on = np.random.uniform(6, 20, npts) * USD_TO_EUR # EUR/MWh, @irenaRenewablePowerGeneration2019 pg 139


    wind_off_global_max = 190 * USD_TO_EUR # EUR/MWh, @irenaRenewablePowerGeneration2019 pg 13
    wind_off_global_min = 66 * USD_TO_EUR
    wind_off_global_avg = 84 * USD_TO_EUR
    variable_wind_off = np.random.uniform(17, 30, npts) * USD_TO_EUR # EUR/MWh, @irenaRenewablePowerGeneration2019 pg 139

    plt.rcParams.update({'font.size': 13})

    averages = {
        'Solar panels (global)': solar_global_avg,
        'Onshore wind (global)': wind_on_global_avg,
        'Offshore wind (global)': wind_off_global_avg
    }
    maxs = {
        'Solar panels (global)': solar_global_max,
        'Onshore wind (global)': wind_on_global_max,
        'Offshore wind (global)': wind_off_global_max
    }
    mins = {
        'Solar panels (global)': solar_global_min,
        'Onshore wind (global)': wind_on_global_min,
        'Offshore wind (global)': wind_off_global_min
    }

    nuclear_capacity = 530 # MW
    nuclear_capfac = rng.normal(0.935, 0.03, npts) # @eiaElectricPowerMonthly
    nuclear_production = nuclear_capacity * nuclear_capfac * HRS_PER_YEAR
    nuclear = cm.cost_nuclear(npts, nuclear_capacity, nuclear_production, rng)

    data = [price_ee['NPS Eesti'], price_global['residential'], os_nocredits, os_noccs, os_ccs_avg, os_ccs_best, gas_noccs, gas_ccs_avg, gas_ccs_best, solar_global_avg, solar_ee, wind_on_global_avg, wind_off_global_avg, nuclear]
    labels = ['Wholesale price (EE)', 'Residential price (global)', 'Oil shale (no credits)', 'Oil shale (no CCS)', 'Oil shale (CCS)', 'Oil shale (next gen CCS)', 'Shale gas (no CCS)', 'Shale gas (CCS)', 'Shale gas (next gen CCS)', 'Solar panels (global)', 'Solar panels (EE)', 'Onshore wind (global)', 'Offshore wind (global)', 'Nuclear']
    pos = [0, 0.3, 1.3, 1.6, 1.9, 2.2, 3.2, 3.5, 3.8, 4.8, 5.1, 6.1, 6.4, 7.4]
    colors = ['grey', 'grey', '#cfb1b1', 'red', 'red', 'red', 'orange', 'orange', 'orange', '#cfb50a', '#cfb50a', '#319fe5', 'blue', 'violet']
    fig, ax = plt.subplots(figsize=(8,8))

    for i, d in enumerate(data):
        if labels[i] in ['Solar panels (global)', 'Onshore wind (global)', 'Offshore wind (global)']:
            boxes = [
                {
                    'whislo': mins[labels[i]],    # Bottom whisker position
                    'q1'    : averages[labels[i]],    # First quartile (25th percentile)
                    'med'   : averages[labels[i]],    # Median         (50th percentile)
                    'q3'    : averages[labels[i]],    # Third quartile (75th percentile)
                    'whishi': maxs[labels[i]],    # Top whisker position
                }
            ]
            bp = ax.bxp(boxes, positions=[pos[i]], widths=0.2, patch_artist=True, showfliers=False)
            plt.setp(bp['medians'], color=colors[i])
        else:
            bp = ax.boxplot(d, positions=[pos[i]], whis=[5,95], widths=0.2, patch_artist=True, showfliers=False)
            if labels[i] == 'Nuclear (Fermi)':
                plt.setp(bp['medians'], color=colors[i])
            else:
                plt.setp(bp['medians'], color='white')

        for element in ['boxes', 'whiskers', 'fliers', 'means', 'caps']:
            plt.setp(bp[element], color=colors[i])

        for patch in bp['boxes']:
            patch.set(facecolor=colors[i])

        plt.setp(bp['whiskers'], linewidth=2.0)

    ax.set_ylim([None, 210])
    ax.set_ylabel('Cost of electricity (EUR/MWh)')
    ax.set_xlabel('')
    ax.set_xticklabels(labels, rotation=90)
    plt.subplots_adjust(bottom=0.3)
    plt.savefig('lcoe_comparison', dpi=400)
    plt.show()


plot_lcoe(10000)
