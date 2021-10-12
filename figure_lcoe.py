"""
Comparison of oil shale + CCS and other electricity production methods
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cost_models as cm
from electric_grid_est import probability_dist

def plot_lcoe(npts):
    rng = np.random.default_rng()
    USD_TO_EUR = 0.876 # OECD data, 2020
    HRS_PER_YEAR = 8760 # h/yr

    years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
    price_list = []
    for y in years:
        price_list.append(pd.read_csv('data/electricity-production and consumption_{}.csv'.format(y), delimiter=';'))
    price_ee = pd.concat(price_list, sort=False)
    price_ee = price_ee.loc[price_ee['NPS Eesti'].notna(), :]
    price_global = pd.read_csv('data/electricity_prices_worldwide.csv', header=3, delimiter=';')

    os_capacity = 600
    os_capfac = rng.normal(0.82, 0.05, npts) # @jamesCostPerformanceBaseline2019
    os_production = os_capacity * os_capfac * HRS_PER_YEAR
    os_noccs, os_ccs_avg, os_ccs_best = cm.cost_os(npts, os_capacity, os_production, rng)
    os_nocredits, _, _ = cm.cost_os(npts, os_capacity, os_production, rng, credits=False)

    gas_capacity = 180
    gas_capfac = rng.normal(0.82, 0.05, npts)
    gas_production = gas_capacity * gas_capfac * HRS_PER_YEAR
    gas_noccs, gas_ccs_avg, gas_ccs_best = cm.cost_os_gas(npts, gas_capacity, gas_production, rng)

    bio_capacity = 600
    bio_capfac = rng.normal(0.82, 0.05, 1)
    bio_production = bio_capacity * bio_capfac * HRS_PER_YEAR
    bio_noccs, bio_ccs_avg, bio_ccs_best = cm.cost_biomass(npts, bio_capacity, bio_production, rng)

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

    # dif = os_ccs_avg - nuclear
    # p = sns.displot(x=dif, kind='kde')
    # p.set(xlim=[-100,130])
    # p.set(xlabel='LCOE erinevus (CCS - tuuma) (EUR/MWh)')
    # p.set(ylabel='Tõenäosus (%)')
    # p.tight_layout()
    # p.savefig('jaotus_ccs_miinus_tuuma.png')
    #
    # prob, values = probability_dist(dif)
    # plt.figure()
    # plt.plot(prob, values)
    # plt.xlim([0,100])
    # plt.ylim([-100,130])
    # plt.xlabel('Tõenäosus (%)')
    # plt.ylabel('LCOE erinevus (CCS - tuuma) (EUR/MWh)')
    # plt.tight_layout()
    # plt.savefig('tõenäosus_ccs_miinus_tuuma.png')
    # plt.show()


    data = [price_ee['NPS Eesti'], price_global['residential'], os_nocredits, os_noccs, os_ccs_avg, os_ccs_best, gas_noccs, gas_ccs_avg, gas_ccs_best, bio_noccs, bio_ccs_avg, bio_ccs_best, solar_global_avg, solar_ee, wind_on_global_avg, wind_off_global_avg, nuclear]
    labels = ['Wholesale price (EST)', 'Residential price (global)', 'Oil shale (no credits)',
        'Oil shale (no CCS)', 'Oil shale (CCS)', 'Oil shale (next gen CCS)',
        'Pyrolysis gas (no CCS)', 'Pyrolysis gas (CCS)', 'Pyrolysis gas (next CCS)',
        'Biomass (no CCS)', 'Biomass (CCS)', 'Biomass (next gen CCS)', 'Solar panels (global)',
        'Solar panels (EST)', 'Onshore wind (global)', 'Offshore wind (global)', 'Nuclear']
    pos = [9.0, 8.7, 7.7, 7.4, 7.1, 6.8, 5.8, 5.5, 5.2, 4.2, 3.9, 3.6, 2.6, 2.3, 1.3, 1.0, 0]
    colors = ['grey', 'grey', '#cfb1b1', 'red', 'red', 'red', 'orange', 'orange', 'orange',
        '#41d055', '#41d055', '#41d055', '#cfb50a', '#cfb50a', '#319fe5', 'blue', 'violet']

    fig, ax = plt.subplots(figsize=(9,10))

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
            bp = ax.bxp(boxes, positions=[pos[i]], widths=0.2, patch_artist=True, showfliers=False, vert=False)
            plt.setp(bp['medians'], color=colors[i])
        else:
            bp = ax.boxplot(d, positions=[pos[i]], whis=[5,95], widths=0.2, patch_artist=True, showfliers=False, vert=False)
            if labels[i] == 'Nuclear (Fermi)':
                plt.setp(bp['medians'], color=colors[i])
            else:
                plt.setp(bp['medians'], color='white')

        for element in ['boxes', 'whiskers', 'fliers', 'means', 'caps']:
            plt.setp(bp[element], color=colors[i])

        for patch in bp['boxes']:
            patch.set(facecolor=colors[i])

        plt.setp(bp['whiskers'], linewidth=2.0)

    ax.set_xlim([None, 210])
    ax.set_xlabel('Cost of electricity (EUR/MWh)')
    ax.set_ylabel('')
    ax.set_yticklabels(labels)
    plt.subplots_adjust(left=0.3)
    plt.tight_layout()
    plt.savefig('figures/lcoe_comparison', dpi=400)
    plt.show()


plot_lcoe(1000000)
