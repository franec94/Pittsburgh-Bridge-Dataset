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


# -----------------------------------------------------------------
# Test with permutations the significance of a classification score
# -----------------------------------------------------------------

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
        plt.legend()
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
    title="significance of classification score", fig_name="significance_of_classification_score.png"):

    """
    Test with permutations the significance of a classification score
    """

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

def try_func_test_significance_of_classification_score():
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
