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


# Imports sklearn
# --------------------------------------------------------------------------- #
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import train_test_split

# Custom Imports
# --------------------------------------------------------------------------- #
from utils_stats.utils_functions import *

# =========================================================================== #
# Functions
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Training Functions
# --------------------------------------------------------------------------- #

def linear_regression_custom(X, y, test_size=0.33, random_state=42, randomize_data: bool = False, n_times: int = 6, scale: float = .1):
    """
    https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols_ridge_variance.html#sphx-glr-auto-examples-linear-model-plot-ols-ridge-variance-py
    """
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    
    # X_train, X_test, y_train, y_test = train_test_split(
    X_train, X_test, y_train, _ = train_test_split( 
        X, y, test_size=test_size, random_state=random_state)
    
    classifiers = dict(ols=linear_model.LinearRegression(),
                   ridge=linear_model.Ridge(alpha=.1))

    for name, clf in classifiers.items():
        fig, ax = plt.subplots(figsize=(4, 3))
        
        if randomize_data:
            for _ in range(n_times):
                this_X = scale * np.random.normal(size=(X_train.shape[0], 1)) + X_train
                clf.fit(this_X, y_train)

                ax.plot(X_test, clf.predict(X_test), color='gray')
                ax.scatter(this_X, y_train, s=3, c='gray', marker='o', zorder=10)
                pass
            pass
        
        clf.fit(X_train, y_train)
        ax.plot(X_test, clf.predict(X_test), linewidth=2, color='blue')
        ax.scatter(X_train, y_train, s=30, c='red', marker='+', zorder=10)

        ax.set_title(name)
        ax.set_xlim(0, 2)
        ax.set_ylim((0, 1.6))
        ax.set_xlabel('X')
        ax.set_ylabel('y')

        fig.tight_layout()
    
    plt.show()
    pass


def classifier_comparison(X, y, start_clf: int = 0, stop_clf: int = 10, verbose: int = 0, record_errors: bool = False) -> object:
    
    assert len(X.shape) == 2, "X must have at list two predictors"
    
    """
    https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
    """
    h = .02  # step size in the mesh

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]
    
    assert start_clf >= 0
    assert stop_clf > 0
    assert start_clf < stop_clf
    assert stop_clf <= len(classifiers)
    
    error_list: list = list()
    
    # rng = np.random.RandomState(2)
    # X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [
        # make_moons(noise=0.3, random_state=0),
        # make_circles(noise=0.2, factor=0.5, random_state=1),
        linearly_separable
    ]

    # _ = plt.figure(figsize=(27, 9)) # figure
    # _ = plt.figure(figsize=(10, 10)) # figure
    _ = plt.figure() # figure
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1
        
        if type(names[start_clf:stop_clf]) is not list:
            names_ = [names[start_clf:stop_clf]]
        else:
            names_= names[start_clf:stop_clf]
        
        if type(classifiers[start_clf:stop_clf]) is not list:
            classifiers_ = [classifiers[start_clf:stop_clf]]
        else:
            classifiers_ = classifiers[start_clf:stop_clf]
                            
        for name, clf in zip(names_, classifiers_):
            # for name, clf in zip(names, classifiers):
            try:
                verbose_message(message=f"Classifier: {name}", verbose=verbose, header_flag=True)
                ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)

                # Plot the decision boundary. For that, we will assign a color to each
                # point in the mesh [x_min, x_max]x[y_min, y_max].
                if hasattr(clf, "decision_function"):
                    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
                else:
                    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

                # Put the result into a color plot
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

                # Plot the training points
                ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                           edgecolors='k')
                # Plot the testing points
                ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                           edgecolors='k', alpha=0.6)

                ax.set_xlim(xx.min(), xx.max())
                ax.set_ylim(yy.min(), yy.max())
                ax.set_xticks(())
                ax.set_yticks(())
                if ds_cnt == 0:
                    ax.set_title(name)
                ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
                i += 1
                pass
            except Exception as err:
                record_error((name, err), error_list=error_list, record_errors=record_errors)
                pass
        pass

    plt.tight_layout()
    plt.show()
    return error_list
