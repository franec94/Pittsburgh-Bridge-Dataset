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
# --------------------------------------------------------------------------- #
import matplotlib as mpl
from matplotlib import pyplot as plt
import chart_studio.plotly.plotly as py

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

# SVMs Classifieres
# --------------------------------------------------------------------------- #
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import svm

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


# =========================================================================== #
# FUNCTIONS
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Training Techniques
# --------------------------------------------------------------------------- #

def get_classifier(type_classifier, params_classifier):
    if type_classifier == 'sgd':
        clf = SGDClassifier(loss=params_classifier.best_params_['clf__loss'], penalty=params_classifier.best_params_['clf__penalty'], \
            alpha=params_classifier.best_params_['clf__alpha'], random_state=70, \
            max_iter=params_classifier.best_params_['clf__max_iter'], tol=None)
    elif type_classifier == 'linear-svm':
        clf = LinearSVC( \
            loss=params_classifier.best_params_['clf__loss'],
            penalty=params_classifier.best_params_['clf__penalty'], \
            C=params_classifier.best_params_['clf__C'], \
            random_state=70, max_iter=50, tol=None)

    elif type_classifier == 'rbf-svm':
        # print(params_classifier.best_params_)
        clf = svm.SVC( \
                kernel=str(params_classifier.best_params_['clf__kernel']), 
                max_iter=int(params_classifier.best_params_['clf__max_iter']), 
                # gamma=float(params_classifier.best_params_['clf__gamma']), \
                gamma=float(0.001),
                C=float(params_classifier.best_params_['clf__C']), \
                random_state=int(70), tol=None)
    elif type_classifier == 'decision-tree':
        clf = DecisionTreeClassifier(random_state=70,
                                     splitter=params_classifier.best_params_['clf__splitter'],
                                     criterion=params_classifier.best_params_['clf__criterion'],
                                     max_features=params_classifier.best_params_['clf__max_features'],
                                    )
    elif type_classifier == 'random-forest':
        clf = RandomForestClassifier(random_state=70,
                                     n_estimators=params_classifier.best_params_['clf__n_estimators'],
                                     criterion=params_classifier.best_params_['clf__criterion'],
                                     bootstrap=params_classifier.best_params_['clf__bootstrap'],
                                    )
    else:
        raise Exception('Error {}'.format(type_classifier))

    print(clf)
    return clf


def evaluate_best_current_model_(X, y, pca, gs_clf, test_size, random_state, type_classifier):
    X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(
                X, y, test_size=50, random_state=random_state)
    X_train_pca_tmp = pca.transform(X_train_tmp)
            
    tmp_clf = get_classifier(type_classifier, gs_clf)

    tmp_clf_trained = tmp_clf.fit(X_train_pca_tmp, y_train_tmp)
    tmp_predicted = tmp_clf_trained.predict(pca.transform(X_test_tmp))

    print("[TRAINING WITH BEST MODEL] --- Classification Report ---")
    print(metrics.classification_report(y_test_tmp, tmp_predicted,
                                                target_names=['negative', 'positive']))

    print("[TRAINING WITH BEST MODEL] --- Confusion Matrix ---")
    print(metrics.confusion_matrix(y_test_tmp, tmp_predicted))
    print(f"{np.mean(tmp_predicted == y_test_tmp)}")
    pass


def grid_search_approach(technique, n, clf, parameters, X, y, test_size, random_state, cv=7, iid=False, n_jobs=-1, sss_flag=False, type_classifier=None):
    '''Performs grid search technique, against a defined classifier or pipeline object and a dictionary of hyper-params.
    
    Params:
    -------
        - n: number or list of numbers, so numbers of principal components to be retained, exploited,
             in order to improve the overall performances.
        
        - clf: scikit-learn Pipeline object, made up of all the operations to be performed in a given order.
        
        - cv: integer, default=7, number to refer to attempt performed by cross-validation technique to create
              cv models picking up their mean.
        
        - iid: boolean, default=False, shows whether input data should be treated as independent and
               identically distributed data samples.
        
        - n_jobs: integer, default=-1, allows, or enables to let the work station within which the training script is lauched to discover
                  and eventually exploit a baunch of cpu for increasing the performance during training phase.
    '''
    # Splitting dataset into training and test sets, respectively, both features and labels
    # -------------------------------------------------------------------------------------
    if sss_flag is False:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        sss.get_n_splits(X, y)

        for train_index, test_index in sss.split(X, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
        # sss = StratifiedShuffleSplit(n_splits=cv, test_size=test_size * 2, random_state=random_state)
        # cv = sss.split(X_train, y_train)

    # Performing only once for all Principal Component
    # -------------------------------------------------------------------------------------
    n_components = X.shape[1]
    pca = PCA(n_components=n_components)
    
    pca = pca.fit(X_train)
    backup_pcs_ = copy.copy(pca.components_)
    # print(f"Shape principal componets: {backup_pcs_.shape}")
    
    print(f'# GRID SEARCH METHOD APPLYED ON: {technique.split(",")[0]} Technique')
    print(f'# PREPROCESSING METHOD: {technique.split(",")[1]} Technique ')
    print('#' + "=" * 100)
    for pos, n_components in enumerate(n):
        print(f"Grid Search attempt no. : {pos+1}")
        print("-" * 100, sep='')
        tmp_cv = cv
        
        if True:
            # Preparing Feature Space by means of retained Principal Components
            # -------------------------------------------------------------------------------------
            n = len(pca.components_[n_components:])    
            pca.components_[n_components:] = [[0] * X.shape[1]] * n
        
            X_train_pca = pca.transform(X_train)

            if sss_flag is True:
                sss = StratifiedShuffleSplit(n_splits=cv, test_size=test_size * 0.5, random_state=random_state)
                cv = sss.split(X_train, y_train)
        
            # Performing training phase
            # -------------------------------------------------------------------------------------
            gs_clf = GridSearchCV(clf, parameters, cv=cv, iid=iid, n_jobs=n_jobs)
            gs_clf = gs_clf.fit(X_train_pca, y_train)
        
        
            # Evaluating performances
            # -------------------------------------------------------------------------------------
            predicted = gs_clf.predict(pca.transform(X_test))
            print()
            print("--- Classification Report ---")
            print(metrics.classification_report(y_test, predicted,
                                                target_names=['negative', 'positive']))

            print("--- Confusion Matrix ---")
            print(metrics.confusion_matrix(y_test, predicted))
            print(f"{np.mean(predicted == y_test)}")
    
            print(f"Best Score: {gs_clf.best_score_}")

            print("--- Best Params ---")
            print(f"n_components: {n_components}")
            for param_name in sorted(parameters.keys()):
                print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
        
            try:
                evaluate_best_current_model_(X, y, pca, gs_clf, test_size, random_state, type_classifier)
                # raise Exception('Ok')
            except Exception as err:
                print(err)
            # evaluate_best_current_model_(X, y, pca, gs_clf, test_size, random_state, type_classifier)
            # Restoring overall pcs for further, subsequent evaluation
            # -------------------------------------------------------------------------------------
            pca.components_ = copy.copy(backup_pcs_)
            cv = tmp_cv
            
        # except Exception as err:
        else: pass #print(err)
    pass


def plot_roc_crossval(X, y):
    n_samples, n_features = X.shape

    # Add noisy features
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # Classification and ROC analysis
    # -----------------------------------------------------------------------------

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=6)
    classifier = svm.SVC(kernel='linear', probability=True,
                         random_state=random_state)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    pass

# --------------------------------------------------------------------------- #
# Classifiers
# --------------------------------------------------------------------------- #
def sgd_classifier_grid_search(X, y,  num_features=None, parameters_sgd_classifier=None):
    test_size, random_state = 0.25, 50

    # Pipeline classifier definition
    clf_sgd = Pipeline([
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=70,
                          max_iter=50, tol=None)),
    ])

    # Hyper-params definition, dictionary containing possible values established
    if parameters_sgd_classifier is None:
        parameters_sgd_classifier = {
            'clf__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'),
            'clf__penalty': ('l2', 'l1', 'elasticnet'),
            'clf__alpha': (1e-1, 1e-2, 1e-3, 1e-4),
            'clf__max_iter': (50, 100, 150, 200, 500, 1000, 1500, 2000, 2500),
            'clf__learning_rate': ('optimal',),
            'clf__tol': (None, 1e-2, 1e-4, 1e-5, 1e-6),
        }

    # Defining number of principal components to be kept, retained for hyper-params selection
    # n = [int(xi) for xi in '2,5,6,7,8,9,10'.split(',')]
    if num_features is None:
        n = [int(xi) for xi in '2,5,6,7,8,9,10'.split(',')]
    else:
        n = num_features

    # Run the grid-search technique
    type_classifier = 'sgd'
    grid_search_approach('SGDClassifier,MinMax', n, clf_sgd, \
        parameters_sgd_classifier, X, y, \
        test_size, random_state, sss_flag=False, \
        type_classifier=type_classifier)
    pass


def svm_linear_classifier_grid_search(X, y, kernel_type=None, num_features=None, parameters_svm=None):
    test_size, random_state = 0.25, 50

    # Pipeline classifier definition
    if kernel_type is None:
        clf_svm = Pipeline([
            ('clf', LinearSVC(loss='squared_hinge', penalty='l2', C=1.0,
                random_state=70, max_iter=50, tol=None)),
        ])
         # Hyper-params definition, dictionary containing possible values established
        if parameters_svm is None:
            parameters_svm = {
                # 'clf__loss': ('hinge', 'squared_hinge'),
                'clf__penalty': ('l2','l1'),
                'clf__C': (1.0, 0.1, 0.001, 0.0001, 10.0),
                # 'clf__max_iter': (50, 100, 150, 200, 500, 1000, 1500, 2000),
                # 'clf__tol': (1e-2, 1e-4, 1e-5, 1e-6),
                # 'clf_dual': (True, False)
            }
        type_classifier = 'linear-svm'
        kernel_type = ''
    
    elif kernel_type == 'svm-rbf-kernel':
        clf_svm = Pipeline([
            ('clf', svm.SVC()),
        ])
        # Hyper-params definition, dictionary containing possible values established
        if parameters_svm is None:
            parameters_svm = {
                'clf__gamma': (0.003, 0.03, 0.05, 0.5, 0.7, 1.0, 1.5),
                'clf__max_iter':(1e+2, 1e+3, 2 * 1e+3, 5 * 1e+3, 1e+4, 1.5 * 1e+3),
                'clf__C': (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10, 1e+2, 1e+3),
            }
        parameters_svm['clf__kernel'] = ('rbf','linear')
        kernel_type = 'RBF_SVM'
        type_classifier = 'rbf-svm'
    else:
        raise Exception('Error')
    
    # Defining number of principal components to be kept, retained for hyper-params selection
    if num_features is None:
        n = [int(xi) for xi in '2,5,6,7,8,9,10'.split(',')]
    else:
        n = num_features

    # Run the grid-search technique
    grid_search_approach('{}_Classifier,MinMax'.format(kernel_type), n, clf_svm, \
        parameters_svm, X, y, \
        test_size, random_state, sss_flag=False, \
        type_classifier=type_classifier)
    pass


def naive_bayes_classifier_grid_search(X, y, num_features=None, parmas_naive_bayes=None):
    
    type_classifier = 'naive-bayes'
    test_size, random_state = 0.20, 50
    
    # Defining number of principal components to be kept, retained for hyper-params selection
    if num_features is None:
        n = [int(xi) for xi in '2,5,6,7,8,9,10'.split(',')]
    else:
        n = num_features
    
    clf_naive_bayes = Pipeline([
        ('clf', GaussianNB()),
    ])
    
    if parmas_naive_bayes is None:
        parmas_naive_bayes = {}
    
    # Run the grid-search technique
    grid_search_approach('{}_Classifier,MinMax'.format('Naive_Bayes'), n, clf_naive_bayes, \
        parmas_naive_bayes, X, y, \
        test_size, random_state, sss_flag=False, \
        type_classifier=type_classifier)
    pass


def decision_tree_classifier_grid_search(X, y, num_features=None, parmas_decision_tree=None):
    
    type_classifier = 'decision-tree'
    test_size, random_state = 0.25, 50
    
    # Defining number of principal components to be kept, retained for hyper-params selection
    if num_features is None:
        n = [int(xi) for xi in '2,5,6,7,8,9,10'.split(',')]
    else:
        n = num_features
    
    clf_decision_tree = Pipeline([
        ('clf', DecisionTreeClassifier(random_state=random_state)),
    ])
    
    if parmas_decision_tree is None:
        parmas_decision_tree = {
            'clf__splitter': ('random', 'best'),
            'clf__criterion':('gini', 'entropy'),
            'clf__max_features': (None, 'auto', 'sqrt', 'log2')
        }
    
    # Run the grid-search technique
    grid_search_approach('{}_Classifier,MinMax'.format('Random_Forest'), n, clf_decision_tree, \
        parmas_decision_tree, X, y, \
        test_size, random_state, sss_flag=False, \
        type_classifier=type_classifier)
    pass


def random_forest_classifier_grid_search(X, y, num_features=None, parmas_random_forest=None):
    
    type_classifier = 'random-forest'
    test_size, random_state = 0.15, 50
    
    # Defining number of principal components to be kept, retained for hyper-params selection
    if num_features is None:
        n = [int(xi) for xi in '2,5,6,7,8,9,10'.split(',')]
    else:
        n = num_features
    
    clf_random_forest = Pipeline([
        ('clf', RandomForestClassifier(random_state=random_state)),
    ])
    
    if parmas_random_forest is None:
        parmas_random_forest = {
            'clf__n_estimators': (3, 5, 7, 10, 30, 50, 70, 100, 150, 200),
            'clf__criterion':('gini', 'entropy'),
            'clf__bootstrap': (True, False)
        }
    
    # Run the grid-search technique
    grid_search_approach('{}_Classifier,MinMax'.format('Random_Forest'), n, clf_random_forest, \
        parmas_random_forest, X, y, \
        test_size, random_state, sss_flag=False, \
        type_classifier=type_classifier)
    pass


# --------------------------------------------------------------------------- #
# Training Custom
# --------------------------------------------------------------------------- #
def fit_all_by_n_components(
    estimators_list, estimators_names,
    X, y,
    n_components=2,
    pca_kernels_list=None, cv_list=None,
    show_plots=False, plot_dest="figures",
    verbose=0):

    assert type(X) is np.ndarray, f"Error: Feature Matrix X's type is not np.ndarray but instead is an instance of type: {type(X)}"
    assert type(y) is np.ndarray, f"Error: target array y's type is not np.ndarray but instead is an instance of type: {type(y)}"

    dfs_list, df = [], None

    if type(estimators_list) is not list:
        estimators_list = [estimators_list]
    # if type(param_grids) is not list: param_grids = [param_grids]
    if type(estimators_names) is not list:
        estimators_names = [estimators_names]

    plot_dest_list = []
    for ii, estimator_name in enumerate(estimators_names[-3:]):
        plot_dest_list.append(os.path.join(plot_dest, estimator_name))
        try: os.makedirs(plot_dest_list[ii])
        except: pass
    pass

    for _, (estimator_obj, estimator_name) in enumerate(zip(estimators_list, estimators_names)):
        res_df1, res_df2 = fit_by_n_components(
            estimator=estimator_obj, \
            X=copy.deepcopy(X), \
            y=copy.deepcopy(y), \
            n_components=n_components, \
            clf_type=f"{estimator_name}", \
            verbose=verbose,
            cv_list=cv_list,
            pca_kernels_list=pca_kernels_list,
            show_plots=show_plots,
            plot_dest=plot_dest_list[ii])
        dfs_list.append(res_df1)
        if df is None:
            df = res_df2
        else:
            df = pd.concat([df,res_df2])
    return dfs_list, df


def fit_by_n_components(
    estimator,
    X, y,
    n_components, clf_type,
    random_state=0, plot_dest="figures",
    pca_kernels_list=None, cv_list=None,
    show_plots=False, show_errors=False,
    verbose=0):
    

    assert type(X) is np.ndarray, f"Error: Feature Matrix X's type is not np.ndarray but instead is an instance of type: {type(X)}"
    assert type(y) is np.ndarray, f"Error: target array y's type is not np.ndarray but instead is an instance of type: {type(y)}"

    data, data_fit_strf = [], []
    
    # print(pca_kernels_list)
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y,
        random_state=random_state)

    kernels_list = ['linear', 'poly', 'rbf', 'cosine',]
    errors_list, plot_dest_list = [], []

    for ii, kernel in enumerate(kernels_list):
        plot_dest_list.append(os.path.join(plot_dest, kernel))
        try: os.makedirs(plot_dest_list[ii])
        except: pass
    pass


    if pca_kernels_list is None:
        pca_kernels_list = ['linear', 'poly', 'rbf', 'cosine',]
    if type(pca_kernels_list) is not list:
        pca_kernels_list = [pca_kernels_list]
    errors_list = []
    
    # print(pca_kernels_list)
    for _, kernel in enumerate(pca_kernels_list):
        step_msg = 'Kernel PCA: {} | {}'.format(kernel.capitalize(), clf_type)
        try:
            if verbose == 1:
                print()
                print('=' * 100)
                print(step_msg)
                print('=' * 100)

            # Prepare data
            Xtrain_transformed, _ = KernelPCA_transform_data(n_components, kernel, Xtrain, Xtest) # Xtest_transformed

            # Perform CV, LOOCV, Stratified CV
            # Once gotten all results exploit them to fill data object list
            # used later to fill in a result dataframe
            data = perform_cv_techniques(data, estimator, Xtrain, Xtrain_transformed, ytrain, cv_list, verbose=0)

            # Perform standard fit and evaluate
            # clf_cloned = sklearn.base.clone(estimator)
            # clf = fit(clf_cloned, Xtrain_transformed, ytrain, Xtest_transformed, ytest, verbose=verbose)
            
            clf_cloned = sklearn.base.clone(estimator)
            clf_cloned.probability=True
            data_fit_strf = fit_strfd(data_fit_strf, kernel, n_components, clf_cloned, X, y, n_splits=2, verbose=0)
            
            if show_plots:
                # show_plots_fit_by_n(clf, kernel, n_components, Xtest_transformed, ytest)
                pass
        except Exception as err:
            # Handle error if one occurs
            print('ERROR: ' + step_msg + ' ' + str(err))
            errors_list.append('ERROR: ' + step_msg + ' ' + str(err))
            pass
        if show_errors:
            # Show error if one occurs, and 'show_errors' flag is set to be True
            print('-' * 100)
            print('Erors')
            print('-' * 100)
            pprint(errors_list)
        
        pass

    # Create and return a dataframe object
    df1 = prepare_output_df(cv_list, pca_kernels_list, data)
    df2 = prepare_output_df_baseline_fit(pca_kernels_list, data_fit_strf, clf_type)
    return df1, df2

# --------------------------------------------------------------------------- #
# Grid Search Custom
# --------------------------------------------------------------------------- #

def grid_search_estimator(estimator, param_grid, X, y, n_components, clf_type, random_state=0, show_plots=False, show_errors=False):
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y,
        random_state=random_state)

    kernels_list = ['linear', 'poly', 'rbf', 'cosine',]
    errors_list = []
    for kernel in kernels_list:
        step_msg = 'Kernel PCA: {} | {}'.format(kernel.capitalize(), clf_type)
        try:
            print()
            print('=' * 100)
            print(step_msg)
            print('=' * 100)
          
            title = 'n_components={} | kernel={}'.format(n_components, kernel)
    
            kernel_pca =KernelPCA( \
                n_components=n_components, \
                kernel=kernel)              
            kernel_pca.fit(Xtrain)
            
            Xtrain_transformed = kernel_pca.transform(Xtrain)
            Xtest_transformed = kernel_pca.transform(Xtest)

            perform_gs_cv_techniques(estimator, param_grid, Xtrain_transformed, ytrain, Xtest_transformed, ytest, title)

            clf = None
            if show_plots:
                plot_roc_curve(
                    clf,
                    Xtest_transformed,
                    ytest,
                    )
                plot_conf_matrix(
                    clf,
                    Xtest_transformed,
                    ytest,
                    title=title)
        except Exception as err:
            err_msg = 'ERROR: ' + step_msg + '- error message: ' + str(err)
            print(err_msg)
            errors_list.append(err_msg)
            pass
        if show_errors:
            print('-' * 100)
            print('Erors')
            print('-' * 100)
            pprint(errors_list)
        pass


def grid_search_all_by_n_components(estimators_list, param_grids, estimators_names, X, y, n_components, pca_kernels_list=None, random_state=0, show_plots=False, show_errors=False, verbose=0, plot_dest="figures", debug_var=False, avoid_func=False):
    # debug_var = False
    if avoid_func is True:
        return None
    plot_dest_list = []
    grid_res_list = []

    if type(estimators_list) is not list:
        estimators_list = [estimators_list]
    if type(param_grids) is not list:
        param_grids = [param_grids]
    if type(estimators_names) is not list:
        estimators_names = [estimators_names]

    if pca_kernels_list is None:
        pca_kernels_list = ['linear', 'poly', 'rbf', 'cosine', 'sigmoid']

    for ii, estimator_name in enumerate(estimators_names[:]):
        plot_dest_list.append(os.path.join(plot_dest, estimator_name))
        try: os.makedirs(plot_dest_list[ii])
        except: pass
    pass
    for ii, (estimator_obj, estimator_name) in enumerate(zip(estimators_list, estimators_names)):
        # pprint(estimator_obj.get_params().keys())
        if verbose == 1 and debug_var is True:
            # print()
            # print('=' * 100)
            print(estimator_name)
            print('=' * 100)
            pass
        if debug_var is False:
            grid_res = grid_search_by_n_components(estimator_obj, param_grids[ii], X, y, n_components, estimator_name, kernels_list=pca_kernels_list, random_state=0, show_plots=False, show_errors=False, verbose=verbose, plot_dest=plot_dest_list[ii], estimator_name=estimator_name)
            grid_res_list.append(grid_res)
    df_grid_searches = prepare_output_df_grid_search(grid_res_list, pca_kernels_list, estimators_names)
    return df_grid_searches


def grid_search_by_n_components(estimator, param_grid, X, y, n_components, clf_type, kernels_list=None, random_state=0, show_plots=False, show_errors=False, verbose=0, plot_dest="figures", estimator_name=None, ignore_func=False):
    # Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=random_state)

    if kernels_list is None:
        kernels_list = ['linear', 'poly', 'rbf', 'cosine', 'sigmoid']
    if ignore_func is True:
        grid_list = [None] * len(kernels_list)
        return grid_list
    errors_list = []
    grid_list = []

    plot_dest_list = []
    for ii, kernel in enumerate(kernels_list):
        plot_dest_list.append(os.path.join(plot_dest, kernel))
        try: os.makedirs(plot_dest_list[ii])
        except: pass
    pass

    for ii, kernel in enumerate(kernels_list):
        step_msg = 'Kernel PCA: {} | {}'.format(kernel.capitalize(), clf_type)
        try:
            if verbose == 1:
                # print()
                # print('=' * 100)
                print(step_msg)
                print('=' * 100)
          
            if estimator_name is not None:
                title = '{} | n_components={} | (PCA) kernel={}'.format(estimator_name, n_components, kernel)
            else:
                title = 'n_components={} | (PCA) kernel={}'.format(n_components, kernel)
    
            # Prepare data
            # Xtrain_transformed, Xtest_transformed = KernelPCA_transform_data(n_components, kernel, Xtrain, Xtest)

            # perform_gs_cv_techniques(estimator, param_grid, Xtrain_transformed, ytrain, Xtest_transformed, ytest, title)
            res_grid, auc, _ = grid_search_stratified_cross_validation( # res_grid, auc, df_list_class_reports
                estimator, param_grid,
                X, y,
                n_components=n_components, kernel=kernel, n_splits=2,
                title=title, show_figures=show_plots,
                plot_dest=plot_dest_list[ii],
                verbose=verbose)
            grid_list.append((res_grid, kernel, auc))

            # if show_plots: pass
        except Exception as err:
            err_msg = 'ERROR: ' + step_msg + '- error message: ' + str(err)
            print(err_msg)
            errors_list.append(err_msg)
            pass
        if show_errors:
            print('-' * 100)
            print('Erors')
            print('-' * 100)
            pprint(errors_list)
        pass
    
    return grid_list
