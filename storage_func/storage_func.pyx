# -*- coding: utf-8 -*-
# setuptools: language=c++

import numpy as np
# cimport storage_func

def storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step):
    return np.asarray(storage_cpp(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step))
    # result = np.asarray(storage_cpp(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step))
    # N = int(result.shape[0] / 2)
    # to_storage = result[:N]
    # stored = result[N:]
    # return to_storage, stored
