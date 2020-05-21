print(__doc__)

# =========================================================================== #
# Imports
# =========================================================================== #

# Critical Imports
# --------------------------------------------------------------------------- #
import warnings; warnings.filterwarnings("ignore")

# Imports through 'from' syntax
# --------------------------------------------------------------------------- #
from itertools import islice
from pprint import pprint
from sklearn import preprocessing

# Standard Imports
# --------------------------------------------------------------------------- #
import copy; import os
import sys; import shutil
import time

# Imports through 'as' syntax
# --------------------------------------------------------------------------- #
import numpy as np; import pandas as pd

# Imports for graphics
# --------------------------------------------------------------------------- #
# Matplotlib pyplot provides plotting API
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import chart_studio.plotly.plotly as py
import seaborn as sns; sns.set()

# --------------------------------------------------------------------------- #
# Utils Functions
# --------------------------------------------------------------------------- #

def record_error(err, error_list: list, record_errors: bool):
    if record_errors is True:
        error_list.append(err)
        pass
    pass


def verbose_message(message: str, verbose: int = 0, header_flag: bool = False) -> None:
    if verbose == 1:
        if header_flag is True:
            new_line: str = '\n'
            a_line : str = ('-' * 100)
            out_msg: str = new_line + a_line + new_line + str(message) + new_line + a_line
            print(out_msg)
        else:
            print(message)
    pass
