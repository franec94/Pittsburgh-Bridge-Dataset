from pprint import pprint

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from pandas.plotting import scatter_matrix
from pandas.plotting import andrews_curves
from pandas.plotting import parallel_coordinates
from pandas.plotting import bootstrap_plot
from pandas.plotting import lag_plot

N_PLOTTING = 3
N_BASIS_PLOTS = 3

# ----------------------------------------------------------------------------------------------------------------------
# show_df_plotting: from pandas.plotting import ...
# ----------------------------------------------------------------------------------------------------------------------

def get_func_list_plotting(order, priority):
    func_list_plotting_tmp = [andrews_curves, parallel_coordinates, bootstrap_plot]
    tmp_list = []
    for ii, p_val in enumerate(priority):
        if p_val == 1:
            tmp_list.append(func_list_plotting_tmp[ii])
            pass
        pass
    func_list_plotting = []
    for _, pos in enumerate(order):
        func_list_plotting.append(tmp_list[pos])
        pass
    return func_list_plotting


def get_axes_show_df_plotting(n, gridshape=None, figsize=(10,10)):
    
    if gridshape == None:
        axes = [None] * n
        return None, axes
    
    nrows_tmp = n // 2 if n % 2 == 0 else n // 2 + 1
    nrows, ncols = gridshape
    
    assert figsize is not None, f'figsize is None'
    # assert (nrows * ncols) == (nrows_tmp * 2), "grid shape is wrong"
    
    fig = plt.figure(figsize=figsize)
    axes = list()
    for ii in range(n):
        axes.append(fig.add_subplot(nrows, ncols, ii+1))
        pass
    
    return fig, axes


def show_df_plotting(df, target_name, gridshape=None, figsize=(10,10), order=None, priority=None):
    
    if order is None:
        order = list(range(N_PLOTTING))
    if priority is None:
        priority = [1 for _ in range(N_PLOTTING)]
    
    func_list_plotting = get_func_list_plotting(order, priority)
    n = len(func_list_plotting)
    
    fig, axes = get_axes_show_df_plotting(n=n, gridshape=gridshape, figsize=figsize)
    
    for ii, a_func_plot in enumerate(func_list_plotting):
        a_func_plot(df, target_name, ax=axes[ii])
    pass



# ----------------------------------------------------------------------------------------------------------------------
# show_df_plotting: basis plots
# ----------------------------------------------------------------------------------------------------------------------


def show_basis_df_plots(df, target_name, gridshape=None, figsize=(10,10)):
    
    n = 3
    fig, axes = get_axes_show_df_plotting(n=n, gridshape=gridshape, figsize=figsize)
    
    pos = 0
    df.plot.hist(alpha=0.5, ax=axes[pos])
    
    pos = pos + 1
    df.plot.box(ax=axes[pos])
    
    pos = pos + 1
    df.plot.area(ax=axes[pos]);
    pass

def show_df_bars(df, target_name, n_elements, columns=None, gridshape=None, figsize=(10,10)):
    
    if columns is None:
        columns = df.columns
    elif type(columns) is not list:
        columns = [columns]
    n = 2
    fig, axes = get_axes_show_df_plotting(n=n, gridshape=gridshape, figsize=figsize)
    
    pos = 0
    df.loc[:n_elements, columns].groupby([target_name]).size().plot(kind='bar', stacked=True, ax=axes[pos])
    
    pos = pos + 1
    df.loc[:n_elements, columns].groupby([target_name]).count().plot(kind='bar', stacked=True, ax=axes[pos])
    pass


def show_df_bars_scaled(df, target_name, n_elements, columns=None, axis=1, gridshape=None, figsize=(10,10)):
    
    # df.loc[:20, [TARGET, 'x']].groupby([TARGET]).count().head(5).apply(lambda x: x*100/sum(x), axis=0).plot(kind="bar", stacked=True)
    # df.loc[:20, [TARGET, 'x']].groupby([TARGET]).groups.keys()
    
    n = 1
    fig, axes = get_axes_show_df_plotting(n=n, gridshape=gridshape, figsize=figsize)
    
    if columns is None:
        columns = df.columns
    elif type(columns) is not list:
        columns = [columns]
    
    pos = 0
    # df.loc[:n_elements, columns].groupby([target_name]).size().plot(kind='bar', stacked=True, ax=axes[pos])
    
    # pos = pos + 1
    rescaled_lambda = lambda x: x*100/sum(x)
    # rescaled_lambda = lambda x: x/sum(x)
    df.loc[:n_elements, columns].groupby([target_name]).count().apply(rescaled_lambda, axis=axis).plot(kind='bar', stacked=True, ax=axes[pos])
    pass