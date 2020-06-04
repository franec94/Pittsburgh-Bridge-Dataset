import sklearn
from pprint import pprint

# Standard Imports (Data Manipulation and Graphics)
import numpy as np    # Load the Numpy library with alias 'np' 
import pandas as pd   # Load the Pandas library with alias 'pd' 

import seaborn as sns # Load the Seabonrn, graphics library with alias 'sns' 

import copy
import os
import sys
from scipy import stats
from scipy import interp
from itertools import islice
from IPython import display
import ipywidgets as widgets
import itertools

# Matplotlib pyplot provides plotting API
import matplotlib as mpl
from matplotlib import pyplot as plt
import chart_studio.plotly.plotly as py

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

# After Training Analysis Imports
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score

# Classifiers Imports
# SVMs Classifieres
from sklearn.svm import LinearSVC
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
from utils.preprocessing_utils import *

def learning_curves_by_kernels(
    estimators_list, estimators_names,
    X, y,
    train_sizes, cv=5,
    n_components=2,
    pca_kernels_list=None, cv_list=None,
    show_plots=False, plot_dest="figures",
    scoring='accuracy',
    verbose=0, by_pairs=False, figsize=(10, 3), savefigs=False, figs_dest='learning_curve', ignore_func=False):

    if ignore_func is True:
        return
    
    try: os.makedirs(figs_dest)
    except: pass

    if type(estimators_list) is not list:
        estimators_list = [estimators_list]

    if type(estimators_names) is not list:
        estimators_names = [estimators_names]

    if pca_kernels_list is None:
        pca_kernels_list = ['linear', 'poly', 'rbf', 'cosine',]
    if type(pca_kernels_list) is not list:
        pca_kernels_list = [pca_kernels_list]

    for _, kernel in enumerate(pca_kernels_list):
        learning_curves_by_components(
            estimators_list[:], estimators_names[:],
            X=copy.deepcopy(X), y=copy.deepcopy(y),
            train_sizes=train_sizes,
            pca_kernel=kernel,
            n_components=n_components,
            verbose=verbose,
            by_pairs=by_pairs,
            scoring=scoring,
            savefigs=savefigs, figs_dest=os.path.join(figs_dest, kernel)
        )
        pass
    pass


def learning_curves_by_components(
    estimators_list, estimators_names,
    X, y,
    train_sizes, cv=5,
    n_components=2,
    pca_kernel='linear', cv_list=None,
    show_plots=False, plot_dest="figures",
    scoring='accuracy',
    verbose=0, by_pairs=False, figsize=(10, 3), title='Learning Curve By Component', savefigs=False, figs_dest=None):

    # assert type(X) is np.ndarray, f"Error: Feature Matrix X's type is not np.ndarray but instead is an instance of type: {type(X)}"
    # assert type(y) is np.ndarray, f"Error: target array y's type is not np.ndarray but instead is an instance of type: {type(y)}"


    try: os.makedirs(os.path.join(figs_dest, scoring))
    except: pass

    if type(estimators_list) is not list:
        estimators_list = [estimators_list]

    if type(estimators_names) is not list:
        estimators_names = [estimators_names]
    
    ax = None
    pos, flag = 0, False
    for ii, (estimator_obj, estimator_name) in enumerate(zip(estimators_list, estimators_names)):
        if verbose == 1:
            print()
            print(estimator_name.capitalize()); print('=' * 100)
        try:
            if by_pairs is True:
                if ii % 2 == 0:
                    fig_name = f"{estimator_name}"
                    if len(estimators_list) % 2 == 1 and len(estimators_list) - 1 == ii: 
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        flag = True
                    else:
                        _, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
                        ax = axs[0]
                else:
                    ax = axs[1]
                    fig_name = f"{fig_name}_{estimator_name}"
            else:
                # fig, axes = plt.subplots(1, 1, figsize=(10, 15))
                fig, axes = plt.subplots(1, 1)
                try:
                    ax = axes[0]
                except: ax = axes

            title = f"Learning Curve: {estimator_name}|#PCs:{n_components}|Pca kernel:{pca_kernel}"
            """
            learning_curves(
                estimator=estimator_obj, \
                X=copy.deepcopy(X),
                y=copy.deepcopy(y),
                n_components=n_components,
                title=title,
                kernel=pca_kernel,
                scoring=scoring,
                train_sizes=train_sizes, cv=cv,
                ax=ax,
            )
            """
            fig_name = f"{estimator_name}"
            Xtrain_transformed_, _ = KernelPCA_transform_data(n_components, pca_kernel, X, None, verbose=0)
            # plot_learning_curve(estimator_obj, title, Xtrain_transformed_, y, axes=ax, ylim=None, cv=5,train_sizes=np.linspace(.1, 1.0, 10))
            plot_learning_curve(estimator_obj, title, Xtrain_transformed_, y, axes=ax, ylim=None, cv=5,train_sizes=train_sizes)
            if ii % 2 == 1 or flag is True:
                fig_name = os.path.join(figs_dest, scoring, f"fig_{pos}_{fig_name}.png")
                pos = pos + 1
                plt.savefig(fig_name)
                pass
            
        except Exception as err:
            print(str(err))
            pass
    if by_pairs is True:
        plt.show()
    pass


def learning_curves(
    estimator,
    X, y,
    train_sizes,
    cv,
    n_components=2, kernel='linear',
    scaler_method='standard', scoring='accuracy', ax=None, title='Learning Curve'):
    
    Xtrain_transformed_, _ = KernelPCA_transform_data(n_components, kernel, X, None, verbose=0)
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, Xtrain_transformed_, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring)

    if scoring == 'neg_log_loss':
        train_scores_mean = -train_scores.mean(axis = 1)
        validation_scores_mean = -validation_scores.mean(axis = 1)
        pass
    else:
        train_scores_mean = train_scores.mean(axis = 1)
        validation_scores_mean = validation_scores.mean(axis = 1)
        pass
    train_sizes, train_scores_mean, validation_scores_mean = \
        get_values_not_nan(train_sizes, train_scores_mean, validation_scores_mean)

    # title = 'Learning curves for a ' + str(estimator).split('(')[0] + ' model'
    if ax is None:
        plt.figure()

        plt.plot(train_sizes, train_scores_mean, label = 'Training error')
        plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

        plt.ylabel(scoring.capitalize(), fontsize = 15)
        plt.xlabel('Training set size', fontsize = 15)
        
        plt.title(title, fontsize = 10, y = 1.03)
        plt.legend()
        plt.show()
    else:
        ax.plot(train_sizes, train_scores_mean, label = 'Training error')
        ax.plot(train_sizes, validation_scores_mean, label = 'Validation error')
        ax.set_ylabel(scoring.capitalize(), fontsize = 15)
        ax.set_xlabel('Training set size', fontsize = 15)
        ax.set_title(title, fontsize = 10, y = 1.03)
        ax.legend()
    pass


def get_values_not_nan(train_sizes, train_scores_mean, validation_scores_mean):
    train_sizes = np.array(list(train_sizes))
    train_scores_mean = np.array(list(train_scores_mean))
    validation_scores_mean = np.array(list(validation_scores_mean))

    def filter_na(val):
        # print(val.shape)
        # pprint(val)
        idxs = np.array(np.where(np.isnan(val)))
        # pprint(idxs)
        idxs_new = list(filter(lambda xi:  xi not in list(idxs[0]), range(len(val))))
        return np.array(idxs_new)

    # print('train scores')
    idx_train = filter_na(train_scores_mean)

    # print('validation scores')
    idx_val = filter_na(validation_scores_mean)

    unique_idxs = np.intersect1d(idx_train, idx_val)
    return train_sizes[unique_idxs], train_scores_mean[unique_idxs], validation_scores_mean[unique_idxs]


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    try:
        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")
    except:
        axes.set_title(title)
        if ylim is not None:
            axes.set_ylim(*ylim)
        axes.set_xlabel("Training examples")
        axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    try:
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
        axes[0].legend(loc="best")
    except:
        axes.grid()
        axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
        axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
        axes.legend(loc="best")
    return
