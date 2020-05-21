import sklearn
from pprint import pprint

# Standard Imports (Data Manipulation and Graphics)
import numpy as np    # Load the Numpy library with alias 'np' 
import pandas as pd   # Load the Pandas library with alias 'pd' 

# sns.set() # Load the Seabonrn, graphics library with alias 'sns'
import seaborn as sns;  sns.set(style="ticks", color_codes=True) 

import copy
from scipy import stats
from scipy import interp
from itertools import islice
import itertools

# Matplotlib pyplot provides plotting API
# --------------------------------------------------------------------------- #
import matplotlib as mpl
from matplotlib import pyplot as plt
import chart_studio.plotly.plotly as py
from matplotlib.colors import ListedColormap

# Preprocessing Imports
# from sklearn.preprocessing import StandardScaler
# --------------------------------------------------------------------------- #
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler # Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import Normalizer     # Normalize data (length of 1)
from sklearn.preprocessing import Binarizer      # Binarization
from sklearn.preprocessing import MinMaxScaler

# Imports for handling Training
# --------------------------------------------------------------------------- #
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

# After Training Analysis Imports
# --------------------------------------------------------------------------- #
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

# --------------------------------------------------------------------------- #
# Classifiers Imports
# --------------------------------------------------------------------------- #

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression    # Parametric Linear Discriminative Model
from sklearn.linear_model import LogisticRegression  # Parametric Linear Discriminative Model


# SVMs Classifieres
# --------------------------------------------------------------------------- #
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.svm import SVC

# Bayesian Classifieres
# --------------------------------------------------------------------------- #
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

# Decision Tree Classifieres
# --------------------------------------------------------------------------- #
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Import scikit-learn classes: Hyperparameters Validation utility functions.
# --------------------------------------------------------------------------- #
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF

# Import scikit-learn classes: model's evaluation step utility functions.
# --------------------------------------------------------------------------- #
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve

from utils.utilities_functions import *
from utils.cross_validation_custom import *
from utils.grid_search_custom import *

from sklearn import datasets

# =========================================================================== #
# Functions
# =========================================================================== #


# --------------------------------------------------------------------------- #
# Training Functions
# --------------------------------------------------------------------------- #

def example_class_report_iris_dataset(avoid_func_flag: bool = False) -> None:

    if avoid_func_flag is True:
        return
    # Loading the Digits dataset
    digits = datasets.load_digits()

    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        pass
    pass

# --------------------------------------------------------------------------- #
# Training Functions
# --------------------------------------------------------------------------- #

def classifier_comparison_by_pca_kernels(
    X, y,
    start_clf: int = 0, stop_clf: int = 10,
    f1=0, f2=1,
    scaler_technique: str = 'StandardScaler',
    straitified_flag: bool = False, kernels_pca_list = None,
    figsize=(27, 9), by_pairs: bool = False, singles: bool = False,
    verbose: int = 0, record_errors: bool = False, avoid_func: bool = False) -> list:

    """
    https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
    """

    if avoid_func is True:
        return list()
    
    if kernels_pca_list is None:
        kernels_pca_list = ['linear', 'poly', 'rbf', 'cosine',]
    elif type(kernels_pca_list) is str:
        if kernels_pca_list not in ['linear', 'poly', 'rbf', 'cosine',]:
            raise f'Error: {kernels_pca_list} not in [linear, poly, rbf, cosine]'
        kernels_pca_list = [kernels_pca_list]

    err_list_all = []
    for kernel_pca in kernels_pca_list:
        err_list = classifier_comparison(
            copy.deepcopy(X), copy.deepcopy(y),
            start_clf=0, stop_clf=10,
            scaler_technique=scaler_technique, straitified_flag=straitified_flag,
            apply_pca_flag=True, kernel_pca=kernel_pca,
            figsize=figsize, by_pairs=by_pairs, singles=singles,
            verbose=0, record_errors=True)
        err_list_all.append((kernel_pca, err_list))
        pass
    return err_list_all

def classifier_comparison(
    X, y,
    start_clf: int = 0, stop_clf: int = 10,
    f1=0, f2=1,
    apply_pca_flag: bool = False, kernel_pca: str = 'linear',
    straitified_flag: bool = False, scaler_technique: str = 'StandardScaler',
    figsize=(27, 9),  by_pairs: bool = False, singles: bool = False,
    verbose: int = 0, record_errors: bool = False,  avoid_func: bool = False) -> object:
    
    """
    https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
    """

    if avoid_func is True:
        return list()
    
    assert len(X.shape) == 2, "X must have at list two predictors"
    
    h = .02  # step size in the mesh
    error_list: list = list()

    names, classifiers = get_classifiers()
    names_, classifiers_ = get_updated_list(names, classifiers, start_clf, stop_clf)
    
    linearly_separable = (X, y)
    datasets = [linearly_separable]

    # _ = plt.figure(figsize=(27, 9)) # figure
    if figsize is None: _ = plt.figure() # figure
    else: _ = plt.figure(figsize=figsize) # figure

    i = 1
    len_dataset, len_classifiers = len(datasets), len(classifiers_)
    ax = manage_figures_shape(len_dataset, len_classifiers, by_pairs, singles, i)

    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds

        X, X_train, X_test, y_train, y_test = manage_data(X, y, straitified_flag, apply_pca_flag, kernel_pca, scaler_technique)
        X_train, X_test = X_train[:, [f1, f2]], X_test[:, [f1, f2]]

        x_min, x_max = X[:, f1].min() - .5, X[:, f1].max() + .5
        y_min, y_max = X[:, f2].min() - .5, X[:, f2].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        show_input_data(ds_cnt, X_train, y_train, X_test, y_test, ax, xx, yy, cm, cm_bright, f1=f1, f2=f2)

        i += 1                    
        for name, clf in zip(names_, classifiers_):
            # for name, clf in zip(names, classifiers):
            try:
                verbose_message(message=f"Classifier: {name}", verbose=verbose, header_flag=True)
                ax = manage_figures_shape(len_dataset, len_classifiers, by_pairs, singles, i)

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
                show_contourf(ds_cnt, X_train, y_train, X_test, y_test, ax, xx, yy, Z, cm, cm_bright, score, title=name, f1=f1, f2=f2)
                i += 1
                pass
            except Exception as err:
                record_error((name, err), error_list=error_list, record_errors=record_errors)
                # raise err
                pass
        pass

    plt.tight_layout()
    plt.show()
    return error_list


def classifier_comparison_grid_search(X, y, start_clf: int = 0, stop_clf: int = 6, figsize=(27, 9), f1=0, f2=1, verbose: int = 0, record_errors: bool = False, apply_pca_flag: bool = False, kernel_pca: str = 'linear', avoid_func: bool = False, straitified_flag: bool = False, by_pairs: bool = False, singles: bool = False) -> object:
    
    if avoid_func is True:
        return list()
    
    assert len(X.shape) == 2, "X must have at list two predictors"
    
    """
    https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
    """
    h = .02  # step size in the mesh

    names, classifiers, params_grid_list = get_classifiers_gs()
    
    error_list: list = list()
    names_, classifiers_ = get_updated_list(names, classifiers, start_clf, stop_clf)
    
    # rng = np.random.RandomState(2)
    # X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [
        # make_moons(noise=0.3, random_state=0),
        # make_circles(noise=0.2, factor=0.5, random_state=1),
        linearly_separable
    ]

    # _ = plt.figure(figsize=(27, 9)) # figure
    if figsize is None:
        _ = plt.figure() # figure
    else:
        _ = plt.figure(figsize=figsize) # figure

    i = 1
    len_dataset, len_classifiers = len(datasets), len(classifiers_)
    ax = manage_figures_shape(len_dataset, len_classifiers, by_pairs, singles, i)

    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds

        X, X_train, X_test, y_train, y_test = manage_data(X, y, straitified_flag, apply_pca_flag, kernel_pca)
        X_train, X_test = X_train[:, [f1, f2]], X_test[:, [f1, f2]]

        x_min, x_max = X[:, f1].min() - .5, X[:, f1].max() + .5
        y_min, y_max = X[:, f2].min() - .5, X[:, f2].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        show_input_data(ds_cnt, X_train, y_train, X_test, y_test, ax, xx, yy, cm, cm_bright, f1=f1, f2=f2)

        i += 1                    
        for ii, (name, clf) in enumerate(zip(names_, classifiers_)):
            # for name, clf in zip(names, classifiers):
            try:
                verbose_message(message=f"Classifier: {name}", verbose=verbose, header_flag=True)
                ax = manage_figures_shape(len_dataset, len_classifiers, by_pairs, singles, i)

                res_grid, auc = grid_search_stratified_cross_validation(
                    clf=clf, param_grid=params_grid_list[ii+start_clf],
                    X=X, y=y,
                    n_components=2, kernel=kernel_pca, n_splits=2,
                    title=f"Classifier: {name}", show_figures=False,
                    verbose=verbose)
                score = res_grid.score(X_test, y_test)

                # Plot the decision boundary. For that, we will assign a color to each
                # point in the mesh [x_min, x_max]x[y_min, y_max].
                if hasattr(clf, "decision_function"):
                    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
                else:
                    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

                # Put the result into a color plot
                Z = Z.reshape(xx.shape)
                show_contourf(ds_cnt, X_train, y_train, X_test, y_test, ax, xx, yy, Z, cm, cm_bright, score, title=name, f1=f1, f2=f2)
                i += 1
                pass
            except Exception as err:
                record_error((name, err), error_list=error_list, record_errors=record_errors)
                # raise err
                pass
        pass

    plt.tight_layout()
    plt.show()
    return error_list

# --------------------------------------------------------------------------- #
# Utils Functions Graphics
# --------------------------------------------------------------------------- #

def show_input_data(ds_cnt, X_train, y_train, X_test, y_test, ax, xx, yy, cm, cm_bright, f1=0, f2=1):
    if ds_cnt == 0:
        if type(ax) is mpl.figure.Figure:
            plt.title("Input data")
            # Plot the training points
            plt.scatter(X_train[:, f1], X_train[:, f2], c=y_train, cmap=cm_bright,
                edgecolors='k')
            # Plot the testing points
            plt.scatter(X_test[:, f1], X_test[:, f2], c=y_test, cmap=cm_bright, alpha=0.6,
                edgecolors='k')
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.xticks(())
            plt.yticks(())
            plt.show()
        else:
            ax.set_title("Input data")
            # Plot the training points
            ax.scatter(X_train[:, f1], X_train[:, f2], c=y_train, cmap=cm_bright,
                edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, f1], X_test[:, f2], c=y_test, cmap=cm_bright, alpha=0.6,
                edgecolors='k')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
    pass


def show_contourf(ds_cnt, X_train, y_train, X_test, y_test, ax, xx, yy, Z, cm, cm_bright, score, title, f1=0, f2=1):
    if type(ax) is mpl.figure.Figure:
        plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        # Plot the training points
        plt.scatter(X_train[:, f1], X_train[:, f2], c=y_train, cmap=cm_bright,
            edgecolors='k')
        # Plot the testing points
        plt.scatter(X_test[:, f1], X_test[:, f2], c=y_test, cmap=cm_bright,
            edgecolors='k', alpha=0.6)

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
        if ds_cnt == 0:
            if type(ax) is mpl.figure.Figure:
                plt.title(title)
        plt.show()
    else:
        # Plot the training points
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        ax.scatter(X_train[:, f1], X_train[:, f2], c=y_train, cmap=cm_bright,
            edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, f1], X_test[:, f2], c=y_test, cmap=cm_bright,
            edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
                
        if ds_cnt == 0:
            if type(ax) is mpl.figure.Figure:
                plt.title(title)
            else:
                ax.set_title(title)
        pass
    pass


def manage_figures_shape(len_dataset, len_classifiers, by_pairs, singles, i):
    if by_pairs:
        # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        n = len_classifiers + 1
        nrows = (n // 2) if n % 2 == 0 else (n // 2 + 1)
        ax = plt.subplot(nrows, 2, i)
    elif singles:
        ax = plt.figure()
    else:
        # ax = plt.subplot(len(datasets), len(classifiers_) + 1, i)
        ax = plt.subplot(len_dataset, len_classifiers + 1, i)
    return ax

# --------------------------------------------------------------------------- #
# Utils Functions
# --------------------------------------------------------------------------- #

def get_classifiers():
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA"
    ]

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
    return names, classifiers


def get_classifiers_gs():

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        # "RBF SVM",
        # "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        # "Neural Net",
        # "AdaBoost",
        # "Naive Bayes",
        # "QDA"
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        # SVC(gamma=2, C=1),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # MLPClassifier(alpha=1, max_iter=1000),
        # AdaBoostClassifier(),
        # GaussianNB(),
        # QuadraticDiscriminantAnalysis()
    ]

    estimators_list = [LogisticRegression(), KNeighborsClassifier(), SGDClassifier(), SVC(), DecisionTreeClassifier(), RandomForestClassifier()]
    estimators_names = ['LogisticRegression', 'KNeighborsClassifier', 'SGDClassifier', 'SVC', 'DecisionTreeClassifier', 'RandomForestClassifier']

    names = estimators_list
    classifiers = estimators_names

    # pca_kernels_list = ['linear', 'poly', 'rbf', 'cosine',]
    # cv_list = [10, 9, 8, 7, 6, 5, 4, 3, 2]

    parmas_logistic_regression = {
        'penalty': ('l1', 'l2', 'elastic'),
        'solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'),
        'fit_intercept': (True, False),
        'tol': (1e-4, 1e-3, 1e-2),
        'C': (1.0, .1, .01, .001),
    }

    parmas_knn_forest = {
        'n_neighbors': (2,3,4,5,6,7,8,9,10),
        'weights': ('uniform', 'distance'),
        'algorithm': ('ball_tree', 'kd_tree', 'brute'),
    }

    parameters_sgd_classifier = {
        'loss': ('log', 'modified_huber'), # ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron')
        'penalty': ('l2', 'l1', 'elasticnet'),
        'alpha': (1e-1, 1e-2, 1e-3, 1e-4),
        'max_iter': (50, 100, 150, 200, 500, 1000, 1500, 2000, 2500),
        'learning_rate': ('optimal',),
        'tol': (None, 1e-2, 1e-4, 1e-5, 1e-6)
    }

    kernel_type = 'svm-rbf-kernel'
    parameters_svm = {
        'gamma': (0.003, 0.03, 0.05, 0.5, 0.7, 1.0, 1.5),
        'max_iter':(1e+2, 1e+3, 2 * 1e+3, 5 * 1e+3, 1e+4, 1.5 * 1e+3),
        # 'penalty': ('l2','l1'),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid',],
        'C': (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10, 1e+2, 1e+3),
        'probability': (True,), 
    }

    parmas_decision_tree = {
        'splitter': ('random', 'best'),
        'criterion':('gini', 'entropy'),
        'max_features': (None, 'auto', 'sqrt', 'log2')
    }

    parmas_random_forest = {
        'n_estimators': (3, 5, 7, 10, 30, 50, 70, 100, 150, 200),
        'criterion':('gini', 'entropy'),
        'bootstrap': (True, False)
    }

    classifiers_params_grid_search = [parmas_logistic_regression, parmas_knn_forest, parameters_sgd_classifier, parameters_svm, parmas_decision_tree, parmas_random_forest]
    return names, classifiers, classifiers_params_grid_search


def get_updated_list(names, classifiers, start_clf, stop_clf):
    assert start_clf >= 0; assert stop_clf > 0
    assert start_clf < stop_clf; assert stop_clf <= len(classifiers)

    if type(names[start_clf:stop_clf]) is not list:
        names_ = [names[start_clf:stop_clf]]
    else:
        names_= names[start_clf:stop_clf]

    if type(classifiers[start_clf:stop_clf]) is not list:
        classifiers_ = [classifiers[start_clf:stop_clf]]
    else:
        classifiers_ = classifiers[start_clf:stop_clf]
    return names_, classifiers_


def performe_X_rescaling(X: np.ndarray, scaler_technique: str = 'StandardScaler') -> np.ndarray:
    _scaler_technique = scaler_technique.lower()
    if _scaler_technique is 'StandardScaler'.lower():
        return StandardScaler().fit_transform(X)
    elif _scaler_technique is 'Normalizer'.lower():
        return Normalizer().fit_transform(X) 
    elif _scaler_technique is 'MinMaxScaler'.lower():
        return MinMaxScaler().fit_transform(X)
    else:
        raise Exception(f"Error: {scaler_technique} is not allowed as rescaling technique")
    pass

def manage_data(X, y, straitified_flag, apply_pca_flag, kernel_pca, scaler_technique='StandardScaler', verbose=0):
    X = performe_X_rescaling(X, scaler_technique)

    straitified_msg = \
        "straitified_flag is True" if straitified_flag is True \
        else "straitified_flag is False"
    apply_pca_msg= \
        "Applied PCA" if straitified_flag is True \
        else "Just Standardized data"

    if straitified_flag is True:
        X_train, X_test, y_train, y_test = get_stratified_groups(X, y)
    else:
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)
        
    if apply_pca_flag is True:
        X_train, X_test, pca = apply_pca(X_train, X_test, n_components=X_train.shape[1], kernel_pca=kernel_pca)
        X = pca.transform(X)
    

    if verbose == 1:
        print(straitified_msg)
        print(apply_pca_msg)

        print("X_train, X_test")
        print(X_train.shape, X_test.shape)
        print(type(X_train), type(X_test))

        print("y_train, y_test")
        print(y_train.shape, y_test.shape)
        print(type(y_train), type(y_test))

    return X, X_train, X_test, y_train, y_test


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


def apply_pca(X_train, X_test, n_components=2, kernel_pca="linear"):
    
    num_features_x_train, num_features_x_test = X_train.shape[1],  X_test.shape[1]
    
    assert n_components > 0
    assert num_features_x_train == num_features_x_test
    assert n_components <= num_features_x_train

    kernels_pca_list = "linear,poly,rbf,sigmoid,cosine,precomputed".split(",")
    assert kernel_pca in kernels_pca_list
    
    pca = KernelPCA(n_components=n_components, kernel=kernel_pca)
    pca = pca.fit(X_train)
    X_pca_train = pca.transform(X_train)
    X_pca_test = pca.transform(X_test)
    
    return X_pca_train, X_pca_test, pca


# =============================================================================================== #
# External Useful Links
# =============================================================================================== #

# sklearn.preprocessing.MinMaxScaler
# ----------------------------------------------------------------------------------------------- #
# - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html