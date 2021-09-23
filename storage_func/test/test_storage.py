import numpy as np
from ..storage_func import storage

def test_empty():
    storage_power = 100
    storage_energy = 100
    eff_charge = 1
    eff_discharge = 0.8
    time_step = 1

    demanded = np.asarray([300.2])
    produced = np.asarray([218.8])
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == 0)

    time_step = 10
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == 0)


def test_almost_empty():
    storage_power = 100
    storage_energy = 100
    eff_charge = 0.9
    eff_discharge = 0.8
    time_step = 1

    demanded = np.asarray([300.0, 300.2])
    produced = np.asarray([305.0, 218.8])
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == np.asarray([5.0, -3.6]))

    time_step = 10
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == np.asarray([5.0, -3.6]))


def test_full():
    storage_power = 100
    eff_charge = 1
    eff_discharge = 0.8
    time_step = 1
    storage_energy = 100/3600 * time_step

    demanded = np.asarray([300.0, 300.0])
    produced = np.asarray([400.0, 301.0])
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == np.asarray([storage_power, 0]))

    time_step = 10
    storage_energy = 100/3600 * time_step
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == np.asarray([storage_power, 0]))


def test_almost_full():
    storage_power = 100
    eff_charge = 1
    eff_discharge = 0.8
    time_step = 1
    storage_energy = 100/3600 * time_step

    demanded = np.asarray([300.0, 300.0])
    produced = np.asarray([390.0, 406.0])
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == np.asarray([90, 10]))

    time_step = 10
    storage_energy = 100/3600 * time_step
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == np.asarray([90, 10]))


def test_normal():
    storage_power = 100
    storage_energy = 100
    eff_charge = 1
    eff_discharge = 0.8
    time_step = 1

    demanded = np.asarray([300.0, 300.0])
    produced = np.asarray([390.0, 270.0])
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == np.asarray([90, -30]))

    time_step = 10
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == np.asarray([90, -30]))


def test_power_limited_charge1():
    storage_power = 100
    storage_energy = 100
    eff_charge = 1
    eff_discharge = 0.8
    time_step = 1

    demanded = np.asarray([300.0])
    produced = np.asarray([430.0])
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == np.asarray([storage_power]))

    time_step = 10
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == np.asarray([storage_power]))


def test_power_limited_charge2():
    storage_power = 100
    storage_energy = 100
    eff_charge = 0.7
    eff_discharge = 0.8
    time_step = 1

    demanded = np.asarray([300.0])
    produced = np.asarray([430.0])
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == np.asarray([storage_power]))

    time_step = 10
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == np.asarray([storage_power]))


def test_power_limited_discharge():
    storage_power = 100
    storage_energy = 100
    eff_charge = 1
    eff_discharge = 0.8
    time_step = 1

    demanded = np.asarray([100, 100, 480.0])
    produced = np.asarray([200, 200, 290.0])
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert to_storage[-1] == -storage_power * eff_discharge

    time_step = 10
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert to_storage[-1] == -storage_power * eff_discharge


def test_discharge_after_full():
    # Test to ensure that discharges are possible after the storage is full
    storage_power = 100
    eff_charge = 1
    eff_discharge = 0.8
    time_step = 1
    storage_energy = 100/3600 * time_step

    demanded = np.asarray([300.0, 400.0, 300.0])
    produced = np.asarray([400.0, 320.0, 310.0])
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == np.asarray([100, -80, 10]))

    time_step = 10
    storage_energy = 100/3600 * time_step
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == np.asarray([100, -80, 10]))


def test_charge_after_empty():
    # Test to ensure that charging occurs as normal after the storage is emptied
    storage_power = 300
    eff_charge = 0.9
    eff_discharge = 0.9
    time_step = 1
    storage_energy = 3600/3600 * time_step

    demanded = np.asarray([300.0, 300.0, 300.0, 300.0])
    produced = np.asarray([311.0, 295.0, 295.0, 309.0])
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == np.asarray([11, -5, -3.9100000000000006, 9]))

    time_step = 10
    storage_energy = 3600/3600 * time_step
    to_storage = storage(demanded, produced, storage_power, storage_energy, eff_charge, eff_discharge, time_step)
    assert np.all(to_storage == np.asarray([11, -5, -3.9100000000000001, 9]))
