import sklearn
from pprint import pprint

# Standard Imports (Data Manipulation and Graphics)
import numpy as np    # Load the Numpy library with alias 'np' 
import pandas as pd   # Load the Pandas library with alias 'pd' 

import seaborn as sns # Load the Seabonrn, graphics library with alias 'sns' 

import copy
from scipy import stats
from scipy import interp
from os import listdir; from os.path import isfile, join
from itertools import islice
from IPython import display
import ipywidgets as widgets
import itertools
import os; import sys

# Matplotlib pyplot provides plotting API
import matplotlib as mpl
from matplotlib import pyplot as plt
import chart_studio.plotly.plotly as py
import matplotlib.image as mpimg

from sklearn import datasets

# Preprocessing Imports
# from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler # Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import Normalizer     # Normalize data (length of 1)
from sklearn.preprocessing import Binarizer      # Binarization

# Imports for handling Training
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import permutation_test_score

# After Training Analysis Imports
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

# Classifiers Imports
# SVMs Classifieres
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn import svm

# Bayesian Classifieres
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

# Decision Tree Classifieres
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

# Import scikit-learn classes: Hyperparameters Validation utility functions.
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve

# Import scikit-learn classes: model's evaluation step utility functions.
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

from utils.utilities_functions import *


# =========================================================================================================================
# Test with permutations the significance of a classification score
# =========================================================================================================================

def show_plot_significance_of_classification_score(permutation_scores, n_classes, pvalue, score, ax=None, save_fig=False, title=None, fig_name=None):

    """
    Show test with permutations the significance of a classification score
    """

    if ax is None:
        plt.hist(permutation_scores, 20, label='Permutation scores',
             edgecolor='black')
        ylim = plt.ylim()
        # BUG: vlines(..., linestyle='--') fails on older versions of matplotlib
        # plt.vlines(score, ylim[0], ylim[1], linestyle='--',
        #          color='g', linewidth=3, label='Classification Score'
        #          ' (pvalue %s)' % pvalue)
        # plt.vlines(1.0 / n_classes, ylim[0], ylim[1], linestyle='--',
        #          color='k', linewidth=3, label='Luck')
        plt.plot(2 * [score], ylim, '--g', linewidth=3,
            label='Classification Score'
            ' (pvalue %s)' % pvalue)
        plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')

        plt.ylim(ylim)
        plt.legend()
        plt.xlabel('Score')
        if title is not None:
            plt.title(title)
            pass
        if save_fig is True:
            assert fig_name is not None, "fig_name parameter should not be None value"
            plt.savefig(fig_name)
            pass
        plt.show()
    else:
        ax.hist(permutation_scores, 20, label='Permutation scores',
             edgecolor='black')
        ylim = ax.get_ylim()
        ax.plot(2 * [score], ylim, '--g', linewidth=3,
            label='Classification Score'
            ' (pvalue %s)' % pvalue)
        ax.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')

        ax.set_ylim(ylim)
        ax.legend()
        ax.set_xlabel('Score')
        if title is not None:
            ax.set_title(title)
            pass
        if save_fig is True:
            assert fig_name is not None, "fig_name parameter should not be None value"
            plt.savefig(fig_name)
            pass
        pass
    pass


def test_significance_of_classification_score(
    X, y, n_classes,
    estimator=SVC(kernel='linear'), cv=StratifiedKFold(2),
    ax=None, verbose=0,
    show_fig=True, save_fig=False,
    title="significance of classification score", fig_name="significance_of_classification_score.png", avoid_func=False):

    """
    Test with permutations the significance of a classification score
    """

    if avoid_func is True: return (None, None, None)

    score, permutation_scores, pvalue = permutation_test_score(
        estimator, X, y, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)

    if verbose == 1:
        print("Classification score %s (pvalue : %s)" % (score, pvalue))

    if show_fig is True:
        # #############################################################################
        # View histogram of permutation scores
        show_plot_significance_of_classification_score(
            permutation_scores,
            n_classes, pvalue,
            score, ax=ax,
            save_fig=save_fig, title=title, fig_name=fig_name
            )
    return score, permutation_scores, pvalue


def try_func_test_significance_of_classification_score(avoid_func=False):
    if avoid_func is True: return
    # #############################################################################
    # Loading a dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    n_classes = np.unique(y).size

    # Some noisy data not correlated
    random = np.random.RandomState(seed=0)
    E = random.normal(size=(len(X), 2200))

    # Add noisy data to the informative features for make the task harder
    X = np.c_[X, E]

    svm = SVC(kernel='linear')
    cv = StratifiedKFold(2)
    test_significance_of_classification_score(
        X, y, n_classes,
        estimator=svm, cv=cv,
        ax=None, verbose=1,
        show_fig=True, save_fig=False,
        title="significance of classification score", fig_name="significance_of_classification_score.png")
    pass


# -----------------------------------------------------------------
# Test with permutations the significance of a classification score
# -----------------------------------------------------------------

def get_kernels(kernels):
    if kernels is None:
        kernels_list = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
    elif type(kernels) is not list:
        kernels_list = [kernels]
    else:
        kernels_list = kernels
        pass
    return kernels_list


def get_axes(default_fig_layout, n, axes, figsize, gridshape=None):
    if default_fig_layout is True and axes is None:
        axes = list()
        fig = plt.figure(figsize=figsize)
        nrows = n // 2 if n % 2 == 0 else n // 2 + 1
        ncols = 2
        for ii in range(n):
            axes.append(fig.add_subplot(nrows, ncols, ii+1))
            pass
        pass
    elif gridshape is not None:
        axes = list()
        fig = plt.figure(figsize=figsize)
        nrows_tmp = n // 2 if n % 2 == 0 else n // 2 + 1
        nrows, ncols = gridshape
        assert (nrows * ncols) == (nrows_tmp * 2), "grid shape is wrong"
        for ii in range(n):
            axes.append(fig.add_subplot(nrows, ncols, ii+1))
            pass
    else:
        axes = [None] * n
        pass
    return axes


def test_significance_of_classification_score_by_kernel_Pca(
    X, y,
    n_classes, n_components,
    estimator=SVC(kernel='linear'), cv=StratifiedKFold(2),
    kernels=None,
    axes=None, verbose=0,
    show_fig=True, save_fig=False,
    default_fig_layout=False,
    gridshape=None,
    figsize=(10, 5),
    title="significance of classification score", fig_name="significance_of_classification_score.png"
    ):


    kernels_list = get_kernels(kernels)

    axes = get_axes(default_fig_layout, len(kernels_list), axes, figsize, gridshape=gridshape)

    for ii, kernel_name in enumerate(kernels_list):

        Xtrain_transformed, _ = KernelPCA_transform_data(n_components=n_components, kernel=kernel_name, Xtrain=X)

        estimator_name = str(estimator).split('(')[0]

        test_significance_of_classification_score(
            Xtrain_transformed, y, n_classes,
            estimator=estimator, cv=cv,
            ax=axes[ii], verbose=verbose,
            show_fig=show_fig, save_fig=save_fig,
            title=f"{estimator_name}|{kernel_name.capitalize()}|Pcs # {n_components}: {title}", fig_name=f"{kernel_name}_{fig_name}")
        pass
    pass


def test_significance_of_classification_score_by_clfs(
    X, y,
    n_classes, n_components,
    estimators,
    cv=StratifiedKFold(2),
    kernels=None,
    axes=None, verbose=0,
    show_fig=True, save_fig=False,
    default_fig_layout=False,
    gridshape=None,
    figsize=(10, 5),
    title="significance of classification score", fig_name="significance_of_classification_score.png"
    ):

    if type(estimators) is not list:
        estimators_list = [estimators]
    else:
        estimators_list = estimators

    for _, estimator in enumerate(estimators_list):
        test_significance_of_classification_score_by_kernel_Pca(
            X, y,
            n_classes=n_classes,
            n_components=n_components,
            estimator=estimator,
            cv=cv,
            kernels=None,
            axes=None, verbose=0,
            default_fig_layout=False,
            figsize=(10, 10),
            gridshape=gridshape,
            show_fig=True, save_fig=False,
            title="Sign. of Class. Score", fig_name="significance_of_classification_score.png"
        )
        pass
    pass
