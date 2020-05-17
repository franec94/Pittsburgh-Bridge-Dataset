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

# --------------------------------------------------------------------------- #
# Confusion Matirx & Roc Curve Custom
# --------------------------------------------------------------------------- #
def plot_conf_matrix(model, Xtest, ytest, title=None):
    
    y_model = model.predict(Xtest)
    mat = confusion_matrix(ytest, y_model)
    
    fig = plt.figure()
    sns.heatmap(mat, square=True, annot=True, cbar=False)
    plt.xlabel('predicted value')
    plt.ylabel('true value')
    if title:
        plt.title(title)
    return fig

def plot_roc_curve_custom(model, X_test, y_test, label, title=None):
    
    y_pred = model.predict_proba(X_test)
    # print('y_test', type(y_test)); print('y_pred', type(y_pred));
    # print('y_test', y_test.shape); print('y_pred', y_pred.shape);
    
    # print('y_test', y_test[0], 'y_pred', y_pred[0])
    
    # y_test_prob = np.array(list(map(lambda xi: [1, 0] if xi == 0 else [0, 1], y_test)))
    # fpr, tpr, _ = roc_curve(y_test_prob, y_pred)
    
    y_pred = np.argmax(y_pred, axis=1)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    
    fig = plt.figure()
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    if  title:
        plt.title('ROC curve: '.format(title))
    else:
        plt.title('ROC curve')
    plt.legend(loc='best')
    # plt.show()
    return fig

def show_plots_fit_by_n(clf, kernel, n_components, Xtest, ytest):
    # Shos some plots if 'show_plot' flag is valued as True
    plot_roc_curve_custom(
        clf,
        Xtest,
        ytest,
        'n_components={} | kernel={}'.format(n_components, kernel))
    plot_conf_matrix(
        clf,
        Xtest,
        ytest,
        title='n_components={} | kernel={}'.format(10, kernel))
    pass

def add_records(data, cv_list, res_kf, res_loo, res_sscv):
    # record = list(map(lambda xi: f"{xi[0]:.2f} (+/-) {xi[1]:.2f}", [xi[1:] for xi in res_kf]))
    record_acc = list(map(lambda xi: f"{xi[1]:.2f}", [xi for xi in res_kf]))
    record_std = list(map(lambda xi: f"(+/-) {xi[2]:.2f}", [xi for xi in res_kf]))
            
    record = list(itertools.chain.from_iterable(list(zip(record_acc, record_std))))
            
    record = record + [f"{res_loo[0]:.2f}"]
    record = record + [f"(+/-) {res_loo[1]:.2f}"]
            
    record = record + [f"{res_sscv[0]:.2f}"]
    record = record + [f"(+/-) {res_sscv[1]:.2f}"]
    # print('len record:', len(record))
    if len(data) == 0:
        data = [[]] * (len(cv_list) + 2)
    for ii in range(0, len(data)):
        # print([record[ii*2], record[ii*2+1]])
        data[ii] = data[ii] + [record[ii*2], record[ii*2+1]]
        # print(f'len data[{ii}]:', len(data[ii]))
        # data.append(copy.deepcopy(record))
        # print(data)
        pass
    return data

def KernelPCA_transform_data(n_components, kernel, Xtrain, Xtest, verbose=0):
    if verbose == 1:
        print('KernelPCA')
        print('-' * 100)
    # Perform kernel PCA
    kernel_pca =KernelPCA( \
        n_components=n_components, \
        kernel=kernel)        
    if verbose == 1:
        print('KernelPCA - Fit')
        print('-' * 100)      
    kernel_pca.fit(Xtrain)                    

    # Transform data accordingly with current Kernel Pca mode
    if verbose == 1:
        print('KernelPCA - Transform')
        print('-' * 100)   
    Xtrain_transformed = kernel_pca.transform(Xtrain)
    Xtest_transformed = kernel_pca.transform(Xtest)

    return Xtrain_transformed, Xtest_transformed

def prepare_output_df(cv_list, pca_kernels_list, data):
    # col_names_acc = list(map(lambda xi: f"ACC(cv={xi})", cv_list))
    # col_names_st = list(map(lambda xi: f"STD(cv={xi})", cv_list))
        
        
    # col_names = list(itertools.chain.from_iterable(list(zip(col_names_acc, col_names_st))))
    # col_names = col_names + ['ACC(loo)', 'STD(loo)', 'ACC(Stfd-CV)', 'STD(Stfd-CV)']
    
    col_names = list(map(lambda xi: f"CV={xi}".lower(), cv_list))
    col_names = col_names + ['loo'.lower(), 'Stfd-CV'.lower()]
    idx_names = copy.deepcopy(col_names)
    
    col_names = []
    for kernel in pca_kernels_list:
        col_names = col_names + [f"{kernel} - ACC".lower().capitalize(), f"{kernel} - STD".lower().capitalize()]
    # df = pd.DataFrame(data=data, columns=col_names,  index=pca_kernels_list)
    # pprint(data)
    # pprint(col_names)
    df = pd.DataFrame(data=data, columns=col_names,  index=idx_names)
    return df

def prepare_output_df_baseline_fit(pca_kernels_list, data, estimator_name):

    col_names = []
    for kernel in pca_kernels_list:
        col_names = col_names + [f"{kernel} - ACC".lower().capitalize(), f"{kernel} - F1".lower().capitalize()]
    df = pd.DataFrame(data=[data], columns=col_names,  index=[estimator_name])
    return df

# --------------------------------------------------------------------------- #
# Utilities Functions Custom Stratified Training and Test Set Creation
# --------------------------------------------------------------------------- #

def get_indices(class_ith_indeces):
    divisor = len(class_ith_indeces) // 2
    max_len = max(len(class_ith_indeces) - divisor, divisor)
    p1a = class_ith_indeces[:max_len]
    p2a = class_ith_indeces[max_len:]
    return [p1a, p2a]

def get_data(p_train, p_test, X, y):
    ytrain_ = [y[ii]for ii in p_train]
    ytest_ = [y[ii]for ii in p_test]
    
    Xtrain_ = [X[ii]for ii in p_train]
    Xtest_ = [X[ii]for ii in p_test]

    assert len(ytrain_) == len(Xtrain_), f"Train {len(ytrain_)} != {len(Xtrain_)} Test {len(ytest_)} ?? {len(Xtest_)}" 
    assert len(ytest_) == len(Xtest_),f"Train {len(ytrain_)} ?? {len(Xtrain_)} Test {len(ytest_)} != {len(Xtest_)}" 
    return Xtrain_, Xtest_, ytrain_, ytest_