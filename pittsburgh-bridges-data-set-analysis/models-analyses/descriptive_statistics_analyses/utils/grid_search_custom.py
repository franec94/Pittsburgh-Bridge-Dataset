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
from tempfile import TemporaryDirectory
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

from sklearn.neighbors import KNeighborsTransformer, KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Import scikit-learn classes: model's evaluation step utility functions.
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

from utils.utilities_functions import *
from utils.display_utils import *
from utils.display_utils import show_C_vs_gamma_params_svm
from utils.sklearn_tests_plot import *


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


def grid_search_stratified_cross_validation(clf, param_grid, X, y, n_components, kernel, n_splits=2, title=None, verbose=0, show_figures=False, plot_dest="figures", flag_no_computation=False):
    # Stratified-K-Fold Cross-Validation
    if verbose == 1:
        # print()
        # print('-' * 100)
        # print('Grid Search | Stratified-K-Fold Cross-Validation')
        # print('-' * 100)
        pass

    Xtrain_, Xtest_, ytrain_, ytest_ = get_stratified_groups(X, y)

    # Prepare data
    Xtrain_transformed_, Xtest_transformed_ = KernelPCA_transform_data(n_components, kernel, Xtrain_, Xtest_, verbose=0)
    
    # skf = StratifiedKFold(n_splits=n_splits)
    # scores = ['precision', 'recall', 'f1']
    scores = ['accuracy']
    grid = None
    df_list = []
    for _ in scores:
        # print("# Tuning hyper-parameters for %s" % score)
        # print()
        grid = GridSearchCV(
            estimator=clf, param_grid=param_grid,
            # scoring=['accuracy', 'f1'],
            scoring=make_scorer(metrics.accuracy_score),
            # scoring='%s_macro' % score,
            # scoring='%s_macro' % score,
            verbose=0) # cv=skf, verbose=0)

        if flag_no_computation is True:
            return None, None, None, None

        grid = GridSearchCV(
            estimator=clf, param_grid=param_grid,
            # scoring=['accuracy', 'f1'],
            scoring=make_scorer(metrics.accuracy_score),
            # scoring='%s_macro' % score,
            # scoring='%s_macro' % score,
            verbose=0) # cv=skf, verbose=0)

        grid.fit(Xtrain_transformed_, ytrain_)
        if verbose == 1:
            # print()
            # print('[*] Best Params:')
            # pprint(grid.best_params_)

            # print()
            # print('[*] Best Estimator:')
            # pprint(grid.best_estimator_)

            # print()
            print('[*] Best Score:')
            pprint(grid.best_score_)
            pass
            # print("Grid scores on development set:")
            # print()

            try:
                means = grid.cv_results_['mean_test_score']
                stds = grid.cv_results_['std_test_score']
                for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                    print("%0.3f (+/-%0.03f) for %r"
                        % (mean, std * 2, params))
                print()
            except: pass
            y_true, y_pred = ytest_, grid.best_estimator_.predict(Xtest_transformed_)
            # print(classification_report(y_true, y_pred))
            # df = from_class_report_to_df(y_true, y_pred, target_names=['class 0', 'class 1'], support=len(y_true))
            df = create_widget_class_report(y_true, y_pred, target_names=['class 0', 'class 1'], support=len(y_true))
            res_clf_report_dict = classification_report(y_true, y_pred, target_names=['class 0', 'class 1'], output_dict=True)
            # print(df)
            display.display(df)
            df_list.append(df)
            # print()
            pass
        pass
    
    # if show_figures is True:
    # fig = plt.figure(figsize=(15, 5))
    # fig = plt.figure(figsize=(10, 5))

    more_plots = type(clf) is sklearn.neighbors.KNeighborsClassifier or type(clf) is sklearn.svm.SVC
    if more_plots is False:
        # fig = plt.figure(figsize=(6, 5))
        fig = plt.figure(figsize=(20, 5))
        conf_matrix_plot_name = os.path.join(plot_dest, "conf_matrix.png")
        plot_conf_matrix(grid.best_estimator_, Xtest_transformed_, ytest_, title=title, plot_name=conf_matrix_plot_name, show_figure=show_figures, ax=fig.add_subplot(1, 3, 1))

        roc_curve_plot_name = os.path.join(plot_dest, "roc_curve.png")
        auc = plot_roc_curve_custom(grid.best_estimator_, Xtest_transformed_, ytest_, title=title, plot_name=roc_curve_plot_name, show_figure=show_figures, ax=fig.add_subplot(1, 3, 2))

        clf = sklearn.clone(grid.best_estimator_)
        test_significance_of_classification_score(
                # Xtest_transformed_, ytest_,
                Xtrain_transformed_, ytrain_,
                n_classes=2,
                estimator=clf,
                cv=StratifiedKFold(2),
                ax=fig.add_subplot(1, 3, 3),
                verbose=1,
                show_fig=True, save_fig=False,
                title="Sign. of Class. Score", fig_name="significance_of_classification_score.png"
            )
        pass
    else:
        if type(clf) is sklearn.neighbors.KNeighborsClassifier:
            compute_k_neighbors_vs_accuracy_wrapper(param_grid, Xtrain_transformed_, ytrain_, ax=None)
            pass
        elif type(clf) is sklearn.svm.SVC:
            # return None, None, None, None
            show_C_vs_gamma_params_svm(Xtrain_transformed_, ytrain_, verbose=verbose, title=f'SVM|Pca-kernel({kernel})')
            pass
        pass

        
        # fig = plt.figure(figsize=(6, 5))
        fig = plt.figure(figsize=(20, 5))
        conf_matrix_plot_name = os.path.join(plot_dest, "conf_matrix.png")
        plot_conf_matrix(grid.best_estimator_, Xtest_transformed_, ytest_, title=title, plot_name=conf_matrix_plot_name, show_figure=show_figures, ax=fig.add_subplot(1, 3, 1))


        roc_curve_plot_name = os.path.join(plot_dest, "roc_curve.png")
        auc = plot_roc_curve_custom(grid.best_estimator_, Xtest_transformed_, ytest_, title=title, plot_name=roc_curve_plot_name, show_figure=show_figures, ax=fig.add_subplot(1, 3, 2))

        clf = sklearn.clone(grid.best_estimator_)
        test_significance_of_classification_score(
            # Xtest_transformed_, ytest_,
            Xtrain_transformed_, ytrain_,
            n_classes=2,
            estimator=clf,
            cv=StratifiedKFold(2),
            ax=fig.add_subplot(1, 3, 3),
            verbose=1,
            show_fig=True, save_fig=False,
            title="Sign. of Class. Score", fig_name="significance_of_classification_score.png"
        )

        pass

    plt.show()

    acc_test = res_clf_report_dict['accuracy']

    return grid, auc, acc_test, df_list


def compute_k_neighbors_vs_accuracy_wrapper(param_grid, Xtrain, ytrain, ax=None):
    
    n_neighbors_list = param_grid['n_neighbors']
    for algorithm in [param_grid['algorithm'][2]]:
        graph_model = KNeighborsTransformer(n_neighbors=max(n_neighbors_list),
                                    algorithm=algorithm,
                                    mode='distance')
        # classifier_model = KNeighborsClassifier(metric='precomputed')
        classifier_model = KNeighborsClassifier(metric='precomputed')

        compute_k_neighbors_vs_accuracy(graph_model, classifier_model, param_grid, Xtrain, ytrain, algorithm_name=algorithm, ax=ax)
        pass
    pass


def compute_k_neighbors_vs_accuracy(graph_model, classifier_model, param_grid, Xtrain, ytrain, algorithm_name, ax=None):

    n_neighbors_list = param_grid['n_neighbors']
    grid_model = None
    with TemporaryDirectory(prefix="sklearn_graph_cache_") as tmpdir:
        full_model = Pipeline(steps=[('graph', graph_model), ('classifier', classifier_model)],
                    memory=tmpdir)
        param_grid_ = copy.deepcopy(param_grid)
        del param_grid_['weights']
        del param_grid_['algorithm']
        del param_grid_['metric']
        del param_grid_['leaf_size']
        keys_new_list = list(map(lambda xi: f"classifier__{xi}", list(param_grid_.keys())))
        param_grid_ = dict(zip(keys_new_list, list(param_grid_.values())))
        grid_model = GridSearchCV(estimator=full_model, param_grid=param_grid_, scoring=make_scorer(metrics.accuracy_score),)
        grid_model.fit(Xtrain, ytrain)
        pass
            
    title = f"{algorithm_name}"
    show_n_neighbors_vs_accuracy(grid_model, n_neighbors_list, title=title, ax=ax)
    pass


def from_class_report_to_df(y_true, y_pred, target_names, support):
    res_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    indeces_df = list(res_report.keys())
    columns_df = list(res_report[list(res_report.keys())[0]].keys())
    data = []
    for _, v in res_report.items():
        record = []
        try:
            for _, v2 in v.items():
                record.append("%.2f" % (v2,))
            data.append(record)
        except:
            record = [""] * 2 + ["%.2f" % (v,)] + ["%d" % (support,)]
            data.append(record)
        pass
    df = pd.DataFrame(data=data, columns=columns_df, index=indeces_df[:])
    return df


def create_widget_class_report(y_true, y_pred, target_names, support):
    df = from_class_report_to_df(y_true, y_pred, target_names=['class 0', 'class 1'], support=len(y_true))
    widget = widgets.Output()
    with widget:
        display.display(df)
    # create HBox
    hbox = widgets.HBox([widget])
    return hbox

# =============================================================================================== #
# Web site links
# =============================================================================================== #
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html