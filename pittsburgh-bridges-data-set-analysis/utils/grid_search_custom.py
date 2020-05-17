import sklearn
from pprint import pprint

# Standard Imports (Data Manipulation and Graphics)
import numpy as np    # Load the Numpy library with alias 'np' 
import pandas as pd   # Load the Pandas library with alias 'pd' 

import seaborn as sns # Load the Seabonrn, graphics library with alias 'sns' 

import copy
from scipy import stats
from scipy import interp
from itertools import islice
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

from utils.utilities_functions import *


def perform_gs_cv_techniques(estimator, param_grid, Xtrain_transformed, ytrain, Xtest_transformed, ytest, title):
    clf_cloned = sklearn.base.clone(estimator)
    grid_search_kfold_cross_validation(clf_cloned, param_grid, Xtrain_transformed, ytrain, Xtest_transformed, ytest, title)
          
    clf_cloned = sklearn.base.clone(estimator)
    grid_search_loo_cross_validation(clf_cloned, param_grid, Xtrain_transformed, ytrain, Xtest_transformed, ytest, title)
                      
    # clf_cloned = sklearn.base.clone(estimator)
    # grid_search_stratified_cross_validation(clf_cloned, param_grid, Xtrain_transformed, ytrain, Xtest_transformed, ytest, n_splits=3, title=title)
    pass

def grid_search_kfold_cross_validation(clf, param_grid, Xtrain, ytrain, Xtest, ytest, title=None):
    # K-Fold Cross-Validation
    print()
    print('-' * 100)
    print('K-Fold Cross Validation')
    print('-' * 100)
    for cv in [3,4,5,10]:
          
        print('#' * 50)
        print('CV={}'.format(cv))
        print('#' * 50)
        clf_cloned = sklearn.base.clone(clf)
        grid = GridSearchCV(
            estimator=clf_cloned, param_grid=param_grid,
            cv=cv, verbose=0)
          
        grid.fit(Xtrain, ytrain)
        print()
        print('[*] Best Params:')
        pprint(grid.best_params_)

        print()
        print('[*] Best Estimator:')
        pprint(grid.best_estimator_)

        print()
        print('[*] Best Score:')
        pprint(grid.best_score_)
          
        plot_conf_matrix(grid, Xtest, ytest, title)
        # plot_roc_curve(grid, Xtest, ytest, label=title, title=title)
        plot_roc_curve(grid, Xtest, ytest)
        pass
    pass

def grid_search_loo_cross_validation(clf, param_grid, Xtrain, ytrain, Xtest, ytest,title=None):
    # Stratified-K-Fold Cross-Validation
    print()
    print('-' * 100)
    print('Stratified-K-Fold Cross-Validation')
    print('-' * 100)

    loo = LeaveOneOut()
    grid = GridSearchCV(
        estimator=clf, param_grid=param_grid,
        cv=loo, verbose=0)
          
    grid.fit(Xtrain, ytrain)
    print()
    print('[*] Best Params:')
    pprint(grid.best_params_)

    print()
    print('[*] Best Estimator:')
    pprint(grid.best_estimator_)

    print()
    print('[*] Best Score:')
    pprint(grid.best_score_)
          
    plot_conf_matrix(grid, Xtest, ytest, title)
    # plot_roc_curve(grid, Xtest, ytest, label=title, title=title)
    plot_roc_curve(grid, Xtest, ytest)
    pass

def grid_search_stratified_cross_validation(clf, param_grid, X, y, n_components, kernel, n_splits=2, title=None, verbose=0, show_figures=False):
    # Stratified-K-Fold Cross-Validation
    if verbose == 1:
        print()
        print('-' * 100)
        print('Grid Search | Stratified-K-Fold Cross-Validation')
        print('-' * 100)

    # Get N-stratified Groups
    class_0_indeces = list(map(lambda val: val[0], filter(lambda val: val[1] == 0, enumerate(y))))
    class_1_indeces = list(map(lambda val: val[0], filter(lambda val: val[1] == 1, enumerate(y))))

    p_class0 = get_indices(class_0_indeces)
    p_class1 = get_indices(class_1_indeces)
    
    # ytrain_ = [y[ii]for ii in p1a] + [y[ii]for ii in p1b] # ytest_ = [y[ii]for ii in p2a] + [y[ii]for ii in p2b]
    p_train = p_class0[0] + p_class1[0]
    p_test = p_class0[1] + p_class1[1]

    Xtrain_, Xtest_, ytrain_, ytest_ = get_data(p_train, p_test, X, y)

    # Prepare data
    Xtrain_transformed_, Xtest_transformed_ = KernelPCA_transform_data(n_components, kernel, Xtrain_, Xtest_, verbose=0)

    

    # skf = StratifiedKFold(n_splits=n_splits)
    grid = GridSearchCV(
        estimator=clf, param_grid=param_grid,
        verbose=0) # cv=skf, verbose=0)
     
    grid.fit(Xtrain_transformed_, ytrain_)
    if verbose == 1:
        print()
        print('[*] Best Params:')
        pprint(grid.best_params_)

        print()
        print('[*] Best Estimator:')
        pprint(grid.best_estimator_)

        print()
        print('[*] Best Score:')
        pprint(grid.best_score_)
        pass
    
    if show_figures is True:
        plot_conf_matrix(grid, Xtest_transformed_, ytest_, title)
        # plot_roc_curve(grid, Xtest, ytest, label=title, title=title)
        plot_roc_curve(grid, Xtest_transformed_, ytest_)
        pass
    pass
