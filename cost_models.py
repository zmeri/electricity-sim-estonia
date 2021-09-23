"""
Comparison of oil shale + CCS and other electricity production methods
"""
import numpy as np
import pandas as pd

DISC_RATE = 0.09 # from economic analysis in KliimaRITA report
USD_TO_EUR = 0.876 # OECD data, 2020

def cost_os(npts, cap, annual_prod, rng, credits=True):
    if cap == 0:
        return 0, 0, 0

    capture_level = 0.9
    co2_os = rng.normal(0.930, 0.060, npts) # tCO2 / MWh, assuming a new oil shale plant with an efficiency of about 43%
    o_and_m = rng.normal(20, 2.5, npts) # EUR / MWh our estimate of the variable cost of producing electricity in an oil shale power plant. We cannot use actual data because Eesti Energia does not publish its production cost. @vabariigivalitsusEestiElektrimajanduseArengukava2006
    if credits:
        co2_credits = rng.normal(40, 15, npts) # cost of the CO2 credits, EUR/tonn CO2
    else:
        co2_credits = 0

    lifespan_os = 35 # years, @jamesCostPerformanceBaseline2019 uses 30 years, but @cuiQuantifyingOperationalLifetimes2019 indicate that often lifetimes are longer (although plant renovations are probably needed)
    construct_time = 4 # years, based on construction time for Auvere oil shale plant, IEA also has data showing 4 years @ieaAveragePowerGeneration2019
    construction_os = rng.normal(1.87e6, 0.15*1.87e6, npts) # EUR / MW, based on construction cost and capacity of Auvere plant, also excluding assumed interest during construction

    trans_storage = rng.normal(25, 5, npts) # EUR/tCO2, estimated cost of transportation and storage. Based on estimates from @metzIPCCSpecialReport2005; @godecPotentialIssuesCosts2017; @hendriksGlobalCarbonDioxide2002; @zepCostsCO2Storage2011
    ccs_avg = rng.normal(63, 10, npts) + trans_storage # EUR/tCO2, estimated cost of capture is based on more than 100 literature estimates, *** kontrollida, mis ole eluiga nendes artiklites
    ccs_best = rng.normal(37, 10, npts) + trans_storage # EUR/tCO2, this estimate includes only next-generation CCS technologies, which have the potential to have a lower cost

    cost_construction = construction_os * cap / construct_time # the initial investment is not discounted, as discussed by @laiLevelizedCostElectricity2017
    for i in range(1, construct_time):
        cost_construction += construction_os * cap / construct_time / (1 + DISC_RATE)**i

    production_os = 0
    os_noccs = np.copy(cost_construction)
    os_ccs_avg = np.copy(cost_construction)
    os_ccs_best = np.copy(cost_construction)
    for i in range(construct_time+1, lifespan_os+1+construct_time):
        os_noccs += (o_and_m + co2_os * co2_credits) * annual_prod / (1 + DISC_RATE)**i
        os_ccs_avg += (o_and_m + (1 - capture_level) * co2_os * co2_credits) * annual_prod / (1 + DISC_RATE)**i
        os_ccs_best += (o_and_m + (1 - capture_level) * co2_os * co2_credits) * annual_prod / (1 + DISC_RATE)**i
        production_os += annual_prod / (1 + DISC_RATE)**i
    os_noccs = os_noccs / production_os
    os_ccs_avg = os_ccs_avg / production_os + ccs_avg * capture_level * co2_os
    os_ccs_best = os_ccs_best / production_os + ccs_best * capture_level * co2_os
    return os_noccs, os_ccs_avg, os_ccs_best

def cost_os_gas(npts, cap, annual_prod, rng):
    if cap == 0:
        return 0, 0, 0

    capture_level = 0.9
    co2_gas = rng.normal(0.2, 0.060, npts) # tCO2 / MWh, @siirdeEstimationCarbonEmission2011
    o_and_m = rng.normal(10, 2.5, npts) # EUR / MWh our estimate of the variable cost of producing electricity in a shale gas power plant. We used the values for an oil shale fired plant as a starting point and subtracted the fuel cost (since shale gas is a byproduct)
    co2_credits = rng.normal(40, 15, npts) # cost of the CO2 credits, EUR/tonn CO2

    lifespan_gas = 35 # years, @jamesCostPerformanceBaseline2019 uses 30 years, but @cuiQuantifyingOperationalLifetimes2019 indicate that often lifetimes are longer (although plant renovations are probably needed)
    construct_time = 3 # years, based on construction time for VKG's gas plant finished in 2015, @virukeemiagruppAastaraamat20152015
    construction_gas = rng.normal(1.6e6, 0.2*1.6e6, npts) # EUR / MW, we expect that the cost for a shale gas plant will be somewhat lower than that of the Auvere plant

    trans_storage = rng.normal(25, 5, npts) # EUR/tCO2, estimated cost of transportation and storage. Based on estimates from @metzIPCCSpecialReport2005; @godecPotentialIssuesCosts2017; @hendriksGlobalCarbonDioxide2002; @zepCostsCO2Storage2011
    ccs_avg = rng.normal(63, 10, npts) + trans_storage # EUR/tCO2, estimated cost of capture is based on more than 100 literature estimates, *** kontrollida, mis ole eluiga nendes artiklites
    ccs_best = rng.normal(37, 10, npts) + trans_storage # EUR/tCO2, this estimate includes only next-generation CCS technologies, which have the potential to have a lower cost

    cost_construction = construction_gas * cap / construct_time # the initial investment is not discounted, as discussed by @laiLevelizedCostElectricity2017
    for i in range(1, construct_time):
        cost_construction += construction_gas * cap / construct_time / (1 + DISC_RATE)**i

    production_gas = 0
    gas_noccs = np.copy(cost_construction)
    gas_ccs_avg = np.copy(cost_construction)
    gas_ccs_best = np.copy(cost_construction)
    for i in range(construct_time+1, lifespan_gas+1+construct_time):
        gas_noccs += (o_and_m + co2_gas * co2_credits) * annual_prod / (1 + DISC_RATE)**i
        gas_ccs_avg += (o_and_m + (1 - capture_level) * co2_gas * co2_credits) * annual_prod / (1 + DISC_RATE)**i
        gas_ccs_best += (o_and_m + (1 - capture_level) * co2_gas * co2_credits) * annual_prod / (1 + DISC_RATE)**i
        production_gas += annual_prod / (1 + DISC_RATE)**i
    gas_noccs = gas_noccs / production_gas
    gas_ccs_avg = gas_ccs_avg / production_gas + ccs_avg * capture_level * co2_gas
    gas_ccs_best = gas_ccs_best / production_gas + ccs_best * capture_level * co2_gas
    return gas_noccs, gas_ccs_avg, gas_ccs_best

def cost_biomass(npts, cap, annual_prod, rng):
    if cap == 0:
        return 0, 0, 0

    capture_level = 0.9
    efficiency = rng.normal(0.42, 0.02, npts) # this would be for a plant producing only electricity. For a combined heat and power plant the efficiency would be lower.
    Hcomb = rng.normal(19, 1, npts) # MJ / kg, lower heating value of wood (dry basis), @sulgCharacterizationDifferentWood2021
    moisture = rng.normal(0.40, 0.05, npts) # moisture content of the biomass fuel, @pedisiusAnalysisWoodChip2021
    Hcomb = Hcomb * (1 - moisture) - 2.969914 * moisture # heating value when taking into account heat lost due to moisture
    mw_carbon = 12.0107 # g / mol
    mw_co2 = 44.0095 # g / mol
    carbon = rng.normal(0.5, 0.02, npts) # carbon content of wood, @sulgCharacterizationDifferentWood2021
    co2_bio = carbon * (1 - moisture) * mw_co2 / mw_carbon / 1000 / (efficiency * Hcomb / 3600)  # tCO2 / MWh
    fuel_cost = rng.normal(17, 4, npts) # EUR / MWh_thermal, based on an expert's estimate
    fuel_cost = fuel_cost / efficiency # EUR / MWh_electric
    o_and_m = rng.normal(10, 2.5, npts) # EUR / MWh our estimate of the operations and maintenance cost of producing electricity in a biomass power plant

    lifespan_bio = 35 # years, @jamesCostPerformanceBaseline2019 uses 30 years, but @cuiQuantifyingOperationalLifetimes2019 indicate that often lifetimes are longer (although plant renovations are probably needed)
    construct_time = 4 # years, based on construction time for Auvere oil shale plant, IEA also has data showing 4 years @ieaAveragePowerGeneration2019
    construction_bio = rng.normal(1.87e6, 0.2*1.87e6, npts) # EUR / MW, based on construction cost and capacity of Auvere plant, also excluding assumed interest during construction

    trans_storage = rng.normal(25, 5, npts) # EUR/tCO2, estimated cost of transportation and storage. Based on estimates from @metzIPCCSpecialReport2005; @godecPotentialIssuesCosts2017; @hendriksGlobalCarbonDioxide2002; @zepCostsCO2Storage2011
    ccs_avg = rng.normal(63, 10, npts) + trans_storage # EUR/tCO2, estimated cost of capture is based on more than 100 literature estimates, *** kontrollida, mis ole eluiga nendes artiklites
    ccs_best = rng.normal(37, 10, npts) + trans_storage # EUR/tCO2, this estimate includes only next-generation CCS technologies, which have the potential to have a lower cost

    cost_construction = construction_bio * cap / construct_time # the initial investment is not discounted, as discussed by @laiLevelizedCostElectricity2017
    for i in range(1, construct_time):
        cost_construction += construction_bio * cap / construct_time / (1 + DISC_RATE)**i

    production_bio = 0
    bio_noccs = np.copy(cost_construction)
    bio_ccs_avg = np.copy(cost_construction)
    bio_ccs_best = np.copy(cost_construction)
    for i in range(construct_time+1, lifespan_bio+1+construct_time):
        bio_noccs += (o_and_m + fuel_cost) * annual_prod / (1 + DISC_RATE)**i
        bio_ccs_avg += (o_and_m + fuel_cost) * annual_prod / (1 + DISC_RATE)**i
        bio_ccs_best += (o_and_m + fuel_cost) * annual_prod / (1 + DISC_RATE)**i
        production_bio += annual_prod / (1 + DISC_RATE)**i
    bio_noccs = bio_noccs / production_bio
    bio_ccs_avg = bio_ccs_avg / production_bio + ccs_avg * capture_level * co2_bio
    bio_ccs_best = bio_ccs_best / production_bio + ccs_best * capture_level * co2_bio
    return bio_noccs, bio_ccs_avg, bio_ccs_best

def cost_solar(npts, cap, annual_prod, rng, include_grid_cost=True):
    lifespan_solar = 25 # years
    capital_solar_ee = rng.lognormal(7.09, 0.16, npts) * 1000 * USD_TO_EUR # EUR/MW, @irenaRenewablePowerGeneration2020 pg 76
    variable_solar = rng.normal(17.8, 1, npts) * 1000 * USD_TO_EUR # EUR/MW/year, @irenaRenewablePowerGeneration2020 pg 81

    # estimate of additional grid costs (see @ueckerdtSystemLCOEWhat2013)
    existing_solar = 40 # MW
    grid_lifespan = 45 # years
    grid_investments = rng.normal(60, 25, npts) # EUR/kW, assumed to be half the value given for wind in @holttinenImpactsLargeAmounts2011 because solar provides more predictable energy (according to @ueckerdtSystemLCOEWhat2013)
    grid_investments[np.where(grid_investments < 0)] = 0
    capital_grid = grid_investments * 1000 * (cap - existing_solar) # any necessary grid improvements have already been made for existing solar installations

    production_solar = 0
    cost_solar = capital_solar_ee * cap
    if include_grid_cost:
        cost_solar += capital_grid
    for i in range(1, lifespan_solar+1):
        cost_solar += variable_solar * cap / (1 + DISC_RATE)**i
        production_solar += annual_prod / (1 + DISC_RATE)**i # formula taken from @irenaRenewablePowerGeneration2019 pg 137

    if include_grid_cost:
        cost_solar -= capital_grid * (grid_lifespan - lifespan_solar) / (1 + DISC_RATE)**lifespan_solar # take into account the salvage value of remaining grid improvements
    solar_ee = cost_solar / production_solar
    return solar_ee

def cost_wind_on(npts, cap, annual_prod, fraction_unused, rng):
    # wind onshore, @irenaRenewablePowerGeneration2019
    if cap == 0:
        return 0

    wind_on = rng.lognormal(3.69803076, 0.26079461, npts) / (1 - fraction_unused) * USD_TO_EUR
    return wind_on

def cost_wind_off(npts, cap, annual_prod, fraction_unused, rng, include_grid_cost=True):
    # Wind offshore, @irenaRenewablePowerGeneration2019
    if cap == 0:
        return 0

    # estimate of additional grid costs (see @ueckerdtSystemLCOEWhat2013)
    grid_lifespan = 45 # years
    grid_investments = rng.normal(125, 50, npts) # EUR/kW, @holttinenImpactsLargeAmounts2011
    grid_investments[np.where(grid_investments < 0)] = 0
    capital_cost = grid_investments * 1000 * cap
    production_wind = 0
    for i in range(grid_lifespan):
        production_wind += annual_prod / (1 + DISC_RATE)**i
    cost_grid = capital_cost / production_wind

    wind_off = rng.lognormal(4.49690421, 0.37057528, npts) / (1 - fraction_unused) * USD_TO_EUR
    wind_off += cost_grid
    return wind_off

def cost_nuclear(npts, cap, annual_prod, rng):
    if cap == 0:
        return 0

    lifespan_nuclear = 60 # years, @dhaeseleerSynthesisEconomicsNuclear2013
    # construct_time = 7 # years, conventional reactor, @ieaWorldEnergyOutlook2006 (see Figure 13.12)
    construct_time = 3 # years, small modular reactor, @abdullaExpertAssessmentsCost2013
    # capital_conv = rng.lognormal(8.3, 0.4, npts) * 1000 # 2020 EUR/MW, conventional reactor, @dhaeseleerSynthesisEconomicsNuclear2013, @loveringHistoricalConstructionCosts2016, @worldnuclearassociationEconomicsNuclearPower
    capital_smr = rng.lognormal(8.39, 0.4, npts) * 1000 # 2020 EUR/MW, small modular reactor, @abdullaExpertAssessmentsCost2013; @kuznetsovCurrentStatusTechnical2011; @vegelEconomicEvaluationSmall2017
    fuel = rng.normal(6, 0.75, npts) * (596.2 / 584.6) # 2020 EUR/MWh @dhaeseleerSynthesisEconomicsNuclear2013 (see also @NuclearPowerEconomics and @rodriguez-penalongaAnalysisCostsSpent2019), CEPCI index used to convert from 2012 to 2020 EUR
    # the above estimate for fuel costs is for the entire fuel life cycle, including waste disposal
    o_and_m = rng.normal(10, 3.5, npts) * (596.2 / 584.6) # 2020 EUR/MWh @dhaeseleerSynthesisEconomicsNuclear2013 (see also @oecdNuclearElectricityGeneration2003), CEPCI index used to convert from 2012 to 2020 EUR
    # according to @oecdNuclearElectricityGeneration2003 decommissioning costs are included in the cost of producing electricity, at least in OECD countries, and so here we consider that they are included in the operating costs
    variable_nuclear = fuel + o_and_m

    cost_nuclear = capital_smr * cap / construct_time # the initial investment is not discounted, as discussed by @laiLevelizedCostElectricity2017
    for i in range(1, construct_time):
        cost_nuclear += capital_smr * cap / construct_time / (1 + DISC_RATE)**i
    production_nuclear = 0
    for i in range(construct_time+1, lifespan_nuclear+1+construct_time):
        cost_nuclear += variable_nuclear * annual_prod / (1 + DISC_RATE)**i
        production_nuclear += annual_prod / (1 + DISC_RATE)**i
    nuclear = cost_nuclear / production_nuclear
    return nuclear

def cost_storage_uphes(npts, energy_cap, power_cap, annual_prod, rng, price, to_storage):
    # underground pumped hydro storage
    if (energy_cap == 0) or (power_cap == 0):
        return 0, 0

    SECONDS_IN_YR = 31536000
    time_step = SECONDS_IN_YR / price.shape[0]

    lifespan_storage = 60 # years, @PaldiskisseKerkibUudne2020
    construct_time = 8 # years, @PaldiskisseKerkibUudne2020

    scaling_factor_energy = 0.85 # @kapilaDevelopmentTechnoeconomicModels2017
    scaling_factor_power = 0.4 # @kapilaDevelopmentTechnoeconomicModels2017
    scale_energy = (6000 / energy_cap)**(1 - scaling_factor_energy)
    scale_power = (500 / power_cap)**(1 - scaling_factor_power)
    capital_power = rng.normal(1200, 200, npts) * 1000 # 2020 EUR/MW, @kapilaDevelopmentTechnoeconomicModels2017; @guoLifeCycleSustainability2020
    capital_energy = rng.normal(30, 7, npts) * 1000 # 2020 EUR/MWh, @kapilaDevelopmentTechnoeconomicModels2017; @guoLifeCycleSustainability2020
    capital_storage = scale_energy * capital_energy * energy_cap + scale_power * capital_power * power_cap # 2020 EUR
    labor_costs = 0.25 * power_cap * 1400 * 12 * 1.33 # number of employees @paidipatiWorkforceDevelopmentHydropower2017, but using typical Estonian salary
    variable_storage = (0.02 * capital_storage + labor_costs) * rng.normal(1, 0.05, npts) # 2020 EUR operating and maintenance costs.

    # potential revenue from energy arbitrage
    arbitrage = np.sum(price * to_storage) * time_step / 3600

    cost_storage = capital_storage / construct_time # the initial investment is not discounted, as discussed by @laiLevelizedCostElectricity2017
    cost_storage_noarb = capital_storage / construct_time # the initial investment is not discounted, as discussed by @laiLevelizedCostElectricity2017
    for i in range(1, construct_time):
        cost_storage += capital_storage / construct_time  / (1 + DISC_RATE)**i
        cost_storage_noarb += capital_storage / construct_time  / (1 + DISC_RATE)**i
    production = 0
    for i in range(construct_time+1, lifespan_storage+1+construct_time):
        cost_storage += (variable_storage - arbitrage) / (1 + DISC_RATE)**i
        cost_storage_noarb += variable_storage / (1 + DISC_RATE)**i
        production += annual_prod / (1 + DISC_RATE)**i

    storage = cost_storage / production
    storage_noarb = cost_storage_noarb / production
    return storage, storage_noarb

def cost_storage_hydrogen(npts, energy_cap, power_cap, annual_prod, rng):
    # hydrogen storage
    if (energy_cap == 0) or (power_cap == 0):
        return 0

    lifespan_storage = 20 # years, @kalinciTechnoeconomicAnalysisStandalone2015
    lifetime_fuel_cell = 7 # years, @kalinciTechnoeconomicAnalysisStandalone2015; @al-sharafiTechnoeconomicAnalysisOptimization2017 -- assumed that fuel cell is used half of the time (the other half the electrolyzer is used to store energy)
    lifetime_electrolyzer = 15 # years, @kalinciTechnoeconomicAnalysisStandalone2015; @al-sharafiTechnoeconomicAnalysisOptimization2017
    lifetime_tank = 20 # years, @kalinciTechnoeconomicAnalysisStandalone2015
    construct_time = 1 # years

    # main cost components: fuel cell, electrolyzer, and hydrogen storage tank @sanghaiTechnoEconomicAnalysisHydrogen2013
    capital_fuel_cell = rng.normal(2900, 650, npts) * 1000 # 2020 EUR/MW,
    capital_electrolyzer = rng.normal(2700, 1600, npts) * 1000 # 2020 EUR/MW,
    capital_tank = rng.normal(27, 14, npts) * 1000 # 2020 EUR/MWh,
    capital_storage = capital_fuel_cell * power_cap + capital_electrolyzer * power_cap + capital_tank * energy_cap # 2020 EUR
    variable_storage = (34.14 * power_cap + 83.34 * power_cap + 0.035 * energy_cap) * rng.normal(1, 0.1, npts) # 2020 EUR operating and maintenance costs. @parissisIntegrationWindHydrogen2011

    cost_storage = capital_storage / construct_time # the initial investment is not discounted, as discussed by @laiLevelizedCostElectricity2017
    for i in range(1, construct_time):
        cost_storage += capital_storage / construct_time  / (1 + DISC_RATE)**i
    production = 0
    for i in range(construct_time+1, lifespan_storage+1+construct_time):
        # add replacement cost if component lifetime is up
        if (i - construct_time - 1) % lifetime_fuel_cell == 0:
            cost_storage += capital_fuel_cell * power_cap / (1 + DISC_RATE)**i
        if (i - construct_time - 1) % lifetime_electrolyzer == 0:
            cost_storage += capital_electrolyzer * power_cap / (1 + DISC_RATE)**i
        if (i - construct_time - 1) % lifetime_tank == 0:
            cost_storage += capital_tank * energy_cap / (1 + DISC_RATE)**i

        cost_storage += variable_storage / (1 + DISC_RATE)**i
        production += annual_prod / (1 + DISC_RATE)**i

    # take into account the salvage cost of remaining equipment
    end_year = lifespan_storage + construct_time
    cost_storage -= capital_fuel_cell * power_cap * (lifetime_fuel_cell - lifespan_storage % lifetime_fuel_cell) / (1 + DISC_RATE)**end_year
    cost_storage -= capital_electrolyzer * power_cap * (lifetime_electrolyzer - lifespan_storage % lifetime_electrolyzer) / (1 + DISC_RATE)**end_year
    cost_storage -= capital_tank * energy_cap * (lifetime_tank - lifespan_storage % lifetime_tank) / (1 + DISC_RATE)**end_year

    storage = cost_storage / production
    return storage
