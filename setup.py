# cython: language_level=3

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize("Ts_iForest_cython.pyx", annotate=True),
    include_dirs = [np.get_include()]
)

# python setup.py build_ext --inplace