# -*- coding: utf-8 -*-
"""
Setup file for compiling to Cython module
"""
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension("storage_func",
        sources=["storage_func.pyx"],
        language="c++")]

setup(name='storage_func',
      description="Function for calculating energy storage in simulations.",
      ext_modules=cythonize(ext_modules, language_level="3"))
