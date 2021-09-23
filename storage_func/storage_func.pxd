# -*- coding: utf-8 -*-
# setuptools: language=c++
from libcpp.vector cimport vector

cdef extern from "storage_base.cpp":
    vector[double] storage_cpp(vector[double] demanded, vector[double] produced, double storage_power,
        double storage_energy, double eff_charge, double eff_discharge, double time_step) except +
