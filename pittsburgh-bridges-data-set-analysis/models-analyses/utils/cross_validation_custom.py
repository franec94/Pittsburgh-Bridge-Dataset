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

def perform_cv_techniques(data, estimator, Xtrain, Xtrain_transformed, ytrain, cv_list, verbose=0):
    # Perform standard CV
    clf_cloned = sklearn.base.clone(estimator)
    res_kf = kfold_cross_validation(clf_cloned, Xtrain_transformed, ytrain, verbose=verbose, cv_list=cv_list)

    # Perform LOOCV
    clf_cloned = sklearn.base.clone(estimator)
    res_loo = loo_cross_validation(clf_cloned, Xtrain_transformed, ytrain, verbose=verbose)
            
    # Perform Stratified Cross Validation
    clf_cloned = sklearn.base.clone(estimator)
    res_sscv = stratified_cross_validation(clf_cloned, Xtrain, ytrain, n_splits=3, verbose=verbose)

    data = add_records(data, cv_list, res_kf, res_loo, res_sscv)

    return data


# --------------------------------------------------------------------------- #
# Cross Validation Custom
# --------------------------------------------------------------------------- #
def kfold_cross_validation(clf, Xtrain, ytrain, Xtest=None, ytest=None, verbose=0, cv_list=[3,4,5,10]):
    # K-Fold Cross-Validation
    if verbose == 1:
        print()
        print('-' * 100)
        print('K-Fold Cross Validation')
        print('-' * 100)
    
    res = []
    for cv in cv_list:
        clf_cloned = sklearn.base.clone(clf)
        scores = cross_val_score(clf_cloned, Xtrain, ytrain, cv=cv)
        if verbose == 1:
            # print("CV=%d | Accuracy: %0.2f (+/- %0.2f)" % (cv, scores.mean(), scores.std() * 2))
            print("CV=%d | Accuracy: %0.2f (+/- %0.2f)" % (cv, scores.mean(), scores.std()))
        res.append([cv, scores.mean(), scores.std() * 2, scores])
    return res

def loo_cross_validation(clf, Xtrain, ytrain, Xtest=None, ytest=None, verbose=0):
    # Leave-One-Out Cross-Validation
    if verbose == 1:
        print()
        print('-' * 100)
        print('Leave-One-Out Cross-Validation')
        print('-' * 100)
    scores = cross_val_score(clf, Xtrain, ytrain, cv=LeaveOneOut())
    if verbose == 1:
        # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    return (scores.mean(), scores.std() * 2, scores)

def stratified_cross_validation(clf, Xtrain, ytrain, Xtest=None, ytest=None, n_splits=3, verbose=0):
    # Stratified-K-Fold Cross-Validation
    if verbose == 1:
        print()
        print('-' * 100)
        print('Stratified-K-Fold Cross-Validation')
        print('-' * 100)
    skf = StratifiedKFold(n_splits=n_splits)
    scores = cross_val_score(clf, Xtrain, ytrain, cv=skf)
    if verbose == 1:
        # print("Accuracy: %0.2f (+/- %0.2f) | Accuracy Test:" % (scores.mean(), scores.std() * 2))
        print("Accuracy: %0.2f (+/- %0.2f) | Accuracy Test:" % (scores.mean(), scores.std()))
    return (scores.mean(), scores.std() * 2, scores)

def fit(clf, Xtrain, ytrain, Xtest=None, ytest=None, verbose=0):
    if verbose == 1:
        print()
        print('-' * 100)
        print('Fit')
        print('-' * 100)
    clf.fit(Xtrain, ytrain)
    y_model = clf.predict(Xtest) # 4. Predict sample's class labels
    if verbose == 1:
        print('accuracy score:', accuracy_score(ytest, y_model))
        print(f"accuracy score (percentage): {accuracy_score(ytest, y_model)*100:.2f}%")
    return clf

def fit_strfd(data_fit_strf, kernel, n_components, clf, X, y, n_splits=2, verbose=0, show_plot=False):
    if verbose == 1:
        print()
        print('-' * 100)
        print('Fit Straitified')
        print('-' * 100)
        pass

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

    # Fit and Predict
    if verbose == 1:
        print()
        print('Fit')
        print('-' * 100)
    clf.fit(Xtrain_transformed_, ytrain_)
    if verbose == 1:
        print()
        print('Predict')
        print('-' * 100)
    y_model = clf.predict(Xtest_transformed_)

    if verbose == 1:
        print('accuracy score:', accuracy_score(ytest_, y_model))
        print(f"accuracy score (percentage): {accuracy_score(ytest_, y_model)*100:.2f}%")

    acc = "%.2f" % (accuracy_score(ytest_, y_model),)
    f1 = "%.2f" %(f1_score(ytest_, y_model, average='macro'), )
    if show_plot is True:
        show_plots_fit_by_n(clf, kernel, n_components, Xtest_transformed_, ytest_)
    # return (clf, acc, (Xtrain_transformed_, ytrain_), (Xtest_transformed_, ytest_))
    data_fit_strf.extend([acc, f1])
    return data_fit_strf
