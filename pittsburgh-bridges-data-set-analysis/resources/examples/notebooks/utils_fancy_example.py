import os; import sys;
from pprint import pprint


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def setup_dir_for_images(dir_path, create_dirs=False, verbose=0):
    if create_dirs is False: return
    
    try:
        os.makedirs(dir_path)
    except Exception as err:
        if verbose == 1:
            print(f'{str(err)}')
        pass
    pass