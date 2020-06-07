from pprint import pprint

import numpy as np
import pandas as pd

import itertools

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# -------------------------------------------------------------------------------------------
# Stem Plot
# -------------------------------------------------------------------------------------------

def show_stem_plot(x, y, xlabel='x', ylabel='y', title='Stem plot', ax=None, fig_name='stem_plot.png', save_fig=False):
    if ax is None:
        line_plot_via_plot(x, y, xlabel=xlabel, ylabel=ylabel, title=title, fig_name=fig_name, save_fig=save_fig)
    else:
        stem_plot_via_ax(x, y, xlabel=xlabel, ylabel=ylabel, title=title, ax=ax) 
    pass

def stem_plot_via_plot(x, y, xlabel='x', ylabel='y', title='Stem plot'):
    # Plot 
    fig=plt.figure()
    plt.stem(x, y, use_line_collection=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_fig is True:
        plt.savefig(fig_name)
    plt.show()
    pass


def stem_plot_via_ax(x, y, xlabel='x', ylabel='y', title='Stem plot', ax=None):
    # Plot 
    ax.stem(x, y, use_line_collection=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    pass

# -------------------------------------------------------------------------------------------
# Line Plot
# -------------------------------------------------------------------------------------------

def show_line_plot(x, y, xlabel='x', ylabel='y', title='Line plot', ax=None, fig_name='line_plot.png', save_fig=False):
    if ax is None:
        line_plot_via_plot(x, y, xlabel=xlabel, ylabel=ylabel, title=title, fig_name=fig_name, save_fig=save_fig)
    else:
        line_plot_via_ax(x, y, xlabel=xlabel, ylabel=ylabel, title=title, ax=ax)    
    pass

def line_plot_via_plot(x, y, xlabel='x', ylabel='y', title='Line plot', fig_name='scatter_plot.png', save_fig=False):
    # Plot 
    fig=plt.figure()
    plt.plot(x, y, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_fig is True:
        plt.savefig(fig_name)
    plt.show()
    pass


def line_plot_via_ax(x, y, xlabel='x', ylabel='y', title='Scatter plot', ax=None):
    # Plot 
    ax.plot(x, y, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    pass

# -------------------------------------------------------------------------------------------
# Scatter Plot
# -------------------------------------------------------------------------------------------

def show_scatter_plot(x, y, xlabel='x', ylabel='y', title='Scatter plot', ax=None, fig_name='scatter_plot.png', save_fig=False):
    if ax is None:
        scatter_plot_via_plot(x, y, xlabel=xlabel, ylabel=ylabel, title=title, fig_name=fig_name, save_fig=save_fig)
    else:
        scatter_plot_via_ax(x, y, xlabel=xlabel, ylabel=ylabel, title=title, ax=ax)    
    pass


def scatter_plot_via_plot(x, y, xlabel='x', ylabel='y', title='Scatter plot', fig_name='scatter_plot.png', save_fig=False):
    # Plot 
    fig=plt.figure()
    plt.scatter(x, y, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_fig is True:
        plt.savefig(fig_name)
    plt.show()
    pass


def scatter_plot_via_ax(x, y, xlabel='x', ylabel='y', title='Scatter plot', ax=None):
    # Plot 
    ax.scatter(x, y, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    pass


# -------------------------------------------------------------------------------------------
# plot_stem_line_scatter Plot
# -------------------------------------------------------------------------------------------

def get_axes_plot_stem_line_scatter(n_plots, gridshape, figsize):
    if gridshape is None:
        axes = [None] * n_plots
        return None, axes
    else:
        axes = []
        nrows, ncols = gridshape
        assert figsize is not None, f'figsize is None'
        fig = plt.figure(figsize=figsize)
        for ii in range(n_plots):
            axes.append(fig.add_subplot(nrows, ncols, ii+1))
            pass
        pass
    return fig, axes


def get_plots_functions(order, n_plots):
    assert len(order) > 0, 'order list is empty'
    # assert len(order) <= n_plots, 'order list is longer than the list of plots'
    plot_func_list_tmp = [show_scatter_plot, show_stem_plot, show_line_plot]
    plot_func_list = [plot_func_list_tmp[xi] for xi in order[:n_plots]]
    return plot_func_list


def plot_stem_line_scatter(x, y, xlabel='x', ylabel='y', title='Plots', fig_name='plot_stem_line_scatter.png', gridshape=None, figsize=(10, 10), order=[0,1,2], save_fig=False):
    n_plots = 3
    
    fig, axes = get_axes_plot_stem_line_scatter(n_plots, gridshape, figsize)
    plot_func_list = get_plots_functions(order, n_plots)
    
    for ii, a_plot_func in enumerate(plot_func_list):
        a_plot_func(x, y, xlabel=xlabel, ylabel=ylabel, ax=axes[ii])
    if save_fig is True and fig is not None:
        fig.savefig(fig_name)
        pass
    pass

# -------------------------------------------------------------------------------------------
# jointplot
# -------------------------------------------------------------------------------------------

def get_axes_sns_plots(n_plots, gridshape, figsize):
    if gridshape is None:
        axes = [None] * n_plots
        return None, axes
    else:
        axes = []
        nrows, ncols = gridshape
        assert figsize is not None, f'figsize is None'
        fig = plt.figure(figsize=figsize)
        for ii in range(n_plots):
            axes.append(fig.add_subplot(nrows, ncols, ii+1))
            pass
        pass
    return fig, axes

def jointplot_df_sns(df, kind='reg', columns=None, gridshape=None, figsize=(10, 10)):
    if columns is None:
        columns = df.columns
    
    n_plots = len(columns) * (len(columns)-1)//2
    fig, axes = get_axes_sns_plots(n_plots, gridshape, figsize)
    
    pairs_columns = []
    for ii, c1 in enumerate(columns[:-1]):
        for jj, c2 in enumerate(columns[ii+1:]):
            pairs_columns.append([c1, c2])
            pass
        pass
    
    for ii, (x, y) in enumerate(pairs_columns):
        sns.jointplot(x=x, y=y, data=df, kind=kind, ax=axes[ii])
    pass


def violinplot_df_sns(df, columns=None, gridshape=None, figsize=(10, 10)):
    if columns is None:
        columns = df.columns
    
    n_plots = len(columns) * (len(columns)-1)//2
    fig, axes = get_axes_sns_plots(n_plots, gridshape, figsize)
    
    pairs_columns = []
    for ii, c1 in enumerate(columns[:-1]):
        for jj, c2 in enumerate(columns[ii+1:]):
            pairs_columns.append([c1, c2])
            pass
        pass
    for ii, (x, y) in enumerate(pairs_columns):
        sns.violinplot(x = x, y = y, data = df, palette = 'rainbow', ax=axes[ii])
    pass


def boxplot_df_sns(df, hue, columns=None, gridshape=None, figsize=(10, 10)):
    if columns is None:
        columns = df.columns
    
    n_plots = len(columns) * (len(columns)-1)//2
    fig, axes = get_axes_sns_plots(n_plots, gridshape, figsize)
    
    pairs_columns = []
    for ii, c1 in enumerate(columns[:-1]):
        for jj, c2 in enumerate(columns[ii+1:]):
            pairs_columns.append([c1, c2])
            pass
        pass
    for ii, (x, y) in enumerate(pairs_columns):
        sns.boxplot(x = x, y = y,  data = df, hue=hue, palette = 'coolwarm', ax=axes[ii])
    pass

