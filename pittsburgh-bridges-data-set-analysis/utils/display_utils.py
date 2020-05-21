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
from sklearn.decomposition import PCA, KernelPCA

# =========================================================================== #
# FUNCTIONS
# =========================================================================== #

def display_heatmap(corr, dest_figures='figures'):
    '''Dispalyes a heatmap related to the correlation matrix computed for the dataset analysed.'''
    f, ax = plt.subplots(figsize=(10, 8))
    
    f.tight_layout()
    ax.set_title("Heatmap whole Preprocessed `Pittsburgh Bridges Data Set` dataset", fontsize=16, fontweight='bold')
    heatmap = sns.heatmap(corr,
                      mask=np.zeros_like(corr, dtype=np.bool),
                      cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
    # loc, labels = plt.xticks()
    _, _ = plt.xticks()

    # for date, row in corr.T.iteritems():
    n = range(corr.shape[0])
    for i, row in zip(n, corr.iterrows()):
        for j, item in enumerate(row[1]):
            # text = heatmap.axes.text(j + 0.5, i + 0.5, round(item, 2),
            _ = heatmap.axes.text(j + 0.5, i + 0.5, round(item, 2),
                                 ha="center", va="center", color="black")


    # heatmap.set_xticklabels(labels, rotation=45)
    heatmap.set_xticklabels(
        heatmap.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    # heatmap.set_yticklabels(labels, rotation=45)
    heatmap.set_yticklabels(
        heatmap.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    
    plot_name = 'heatmap_corr_matrix.png'
    try: os.makedirs(dest_figures)
    except: pass
    # heatmap.savefig(os.path.join(dest_figures, plot_name))
    plt.savefig(os.path.join(dest_figures, plot_name))

    plt.show()
    pass

def show_cum_variance_vs_components(pca, n_components):
    # tot = sum(pca.explained_variance_)
    _ = sum(pca.explained_variance_)

    var_exp = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(pca.explained_variance_ratio_)


    trace1 = dict(
        type='bar',
        x=['PC %s' %i for i in range(1,n_components+1)],
        y=var_exp,
        name='Individual'
    )

    trace2 = dict(
        type='scatter',
        x=['PC %s' %i for i in range(1,n_components+1)], 
        y=cum_var_exp,
        name='Cumulative'
    )

    data = [trace1, trace2]

    layout=dict(
        title='Explained variance by different principal components',
        yaxis=dict(
            title='Explained variance in percent'
            ),
        xaxis=dict(
            title='number of components'
            ),
        annotations=list([
            dict(
                x=1.16,
                y=1.05,
                xref='paper',
                yref='paper',
                text='Explained Variance',
                showarrow=False,
                )
            ])
    )

    # pass
    # fig = dict(data=data, layout=layout)
    # py.sign_in('franec94', 'QbLNKpC0EZB0kol0aL2Z')
    # py.iplot(fig, filename='selecting-principal-components')

    return dict(data=data, layout=layout)

def show_frequency_distribution_predictors(df, columns_2_avoid=None, features_vs_values=None):
    
    if columns_2_avoid is not None:
        columns_2_keep = list(filter(lambda x: x not in columns_2_avoid, df.columns))
    else:
        columns_2_keep = df.columns
    
    sns.set(style="darkgrid")
    # for index, predictor in enumerate(df.columns):
    for _, predictor in enumerate(columns_2_keep):
        # print(index, predictor)
        predictor_count = df[predictor].value_counts()

        if features_vs_values is not None:
            l = list()
            print(features_vs_values[predictor])
            for k, v in features_vs_values[predictor].items():
                for val in predictor_count.index:
                    if val == v:
                        l.append(k)
                        break
            sns.barplot(l, predictor_count.values, alpha=0.9)
        else:
            sns.barplot(predictor_count.index, predictor_count.values, alpha=0.9)
        
        plt.title('Frequency Distribution of %s' % (predictor))
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('%s' % (predictor), fontsize=12)
        plt.show()
    pass

def build_boxplot(df, predictor_name=None, columns_2_avoid=None, features_vs_values=None, target_col=None):
    
    fig, ax = plt.subplots()
    
    # Setu up columns names to be used for building up related histograms
    if columns_2_avoid is not None:
        # if 'columns_2_avoid' is not None filter those columns
        columns_2_keep = list(filter(lambda x: x not in columns_2_avoid, df.columns))
    else:
        columns_2_keep = df.columns

    if predictor_name is not None:
        if type(predictor_name) is not str:
            # if 'redicotr_name' is not str
            # treat it as a iterable
            columns_2_keep = list(set(predictor_name) & set(columns_2_keep))
        else:
            columns_2_keep = list(set([predictor_name]) & set(columns_2_keep))
    
    sns.set(style="darkgrid")
    
    target = df[target_col].value_counts()
    target_vals = df[target_col].values
    target_idx = target.index
    # target_vals = target.values
    # for index, predictor in enumerate(df.columns):
    for _, predictor in enumerate(columns_2_keep):
        # print(index, predictor)
        # predictor_count = df[predictor].value_counts()
        predictor_vals = df[predictor].values
        
        data = []
        for _, idx in enumerate(target_idx):
            
            res = [i for i, val in enumerate(target_vals == idx) if val]
            vals = [predictor_vals[i] for i, val in enumerate(res)]
            data.append(vals)
            pass
    
        # build a box plot
        ax.boxplot(data)
        ax.set_title('box plot')

        xticklabels = list(map(lambda xi: str(xi), target_idx))
        ax.set_xticklabels(xticklabels)

        # show the plot
        plt.show()
        pass
    pass

def show_frequency_distribution_predictor(df, predictor_name=None, columns_2_avoid=None, features_vs_values=None, target_col=None):
    
    # Setu up columns names to be used for building up related histograms
    if columns_2_avoid is not None:
        # if 'columns_2_avoid' is not None filter those columns
        columns_2_keep = list(filter(lambda x: x not in columns_2_avoid, df.columns))
    else:
        columns_2_keep = df.columns

    if predictor_name is not None:
        if type(predictor_name) is not str:
            # if 'redicotr_name' is not str
            # treat it as a iterable
            columns_2_keep = list(set(predictor_name) & set(columns_2_keep))
        else:
            columns_2_keep = list(set([predictor_name]) & set(columns_2_keep))
    
    sns.set(style="darkgrid")
    # for index, predictor in enumerate(df.columns):
    for _, predictor in enumerate(columns_2_keep):
        # print(index, predictor)
        predictor_count = df[predictor].value_counts()

        if features_vs_values is not None:
            l = list()
            print(features_vs_values[predictor])
            for k, v in features_vs_values[predictor].items():
                for val in predictor_count.index:
                    if val == v:
                        l.append(k)
                        break
            sns.barplot(l, predictor_count.values, alpha=0.9)
        else:
            sns.barplot(predictor_count.index, predictor_count.values, alpha=0.9)
        
        plt.title('Frequency Distribution of %s' % (predictor))
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('%s' % (predictor), fontsize=12)
        plt.show()
    pass

def show_histograms_from_heatmap_corr_matrix(corr_matrix, num_rows=None, row_names=None):
    assert type(corr_matrix) is pd.DataFrame, f"corr_matrix's type is {type(corr_matrix)}, that is not of type pd.DataFrame as requested"
    if num_rows is None:
        num_rows = corr_matrix.shape[0]
    if row_names is None:
        row_names = list(map(lambda xi: 'Variable no.%d' % (xi,), range(corr_matrix.shape[0])))
    pairs = zip(range(num_rows), row_names[:num_rows])
    for _, (ii, row_name) in enumerate(pairs):
        plt.figure()
        plt.title(row_name)
        plt.hist(corr_matrix.values[ii])
        pass
    pass

def show_categorical_predictor_values(df, columns_2_avoid=None, verbose=0):
    
    if columns_2_avoid is not None:
        columns_2_keep = list(filter(lambda x: x not in columns_2_avoid, df.columns))
    else:
        columns_2_keep = df.columns
        
    max_len_name = max(list(map(lambda xi: len(xi), columns_2_keep)))
    
    list_columns = list()
    # for index, predictor in enumerate(df.columns):
    for _, predictor in enumerate(columns_2_keep):
        # print(index, predictor)
        labels = df[predictor].astype('category').cat.categories.tolist()
        # pprint.pprint(predictor, labels)
        if verbose == 1:
            print(f"%-{max_len_name}s" % (predictor,), ':', labels)
        if '?' in labels:
            list_columns.append(predictor)

    # pass
    return list_columns

def display_cumulative_variance_dataset(X, scaler_method=None):
    n_components = X.shape[1]
    pca = PCA(n_components=n_components)
    # pca = PCA(n_components=2)

    #X_pca = pca.fit_transform(X)
    pca = pca.fit(X)
    _ = pca.transform(X)
    fig = show_cum_variance_vs_components(pca, n_components)

    py.sign_in('franec94', 'QbLNKpC0EZB0kol0aL2Z')
    if scaler_method is None:
        py.iplot(fig, filename='selecting-principal-components')
        # py.plot(fig, filename='selecting-principal-components')
    else:
        py.iplot(fig, filename='selecting-principal-components {}'.format(scaler_method))
        # py.plot(fig, filename='selecting-principal-components {}'.format(scaler_method))
    
    principal_components = [pc for pc in '2,5,6,7,8,9,10'.split(',')]
    for _, pc in enumerate(principal_components):
        n_components = int(pc)
    
        cum_var_exp_up_to_n_pcs = np.cumsum(pca.explained_variance_ratio_)[n_components-1]
        print(f"Cumulative varation explained up to {n_components} pcs = {cum_var_exp_up_to_n_pcs}")
    pass

def show_pca_1_vs_pca_2_pcaKernel(X, pca_kernels_list, target_col, dataset, n_components=2):
    # pca_kernels_list = ['linear', 'poly', 'rbf', 'cosine',]
    for ii, kernel in enumerate(pca_kernels_list):
        plt.figure()
        model = KernelPCA(n_components=n_components, kernel=kernel)
        model.fit(X)              
        X_2D = model.transform(X)

        df = pd.DataFrame()
        df['PCA1'] = X_2D[:, 0]
        df['PCA2'] = X_2D[:, 1]
        df[target_col] = dataset[target_col].values

        sns.lmplot("PCA1", "PCA2", hue=target_col, data=df, fit_reg=False)
    pass

def show_scatter_plots_pcaKernel(X, pca_kernels_list, target_col, dataset, n_components, dest_dir='figures'):
    # pca_kernels_list = ['linear', 'poly', 'rbf', 'cosine',]
    for ii, kernel in enumerate(pca_kernels_list):
        plt.figure()
        model = KernelPCA(n_components=n_components, kernel=kernel)
        model.fit(X)              
        X_t = model.transform(X)
    
        col_names = list(map(lambda xi: f"PCA{xi+1}", range(n_components)))
        df = pd.DataFrame(data=X_t, columns=col_names)
        df[target_col] = dataset[target_col].values

        # sns.lmplot("PCA1", "PCA2", hue=target_col, data=df, fit_reg=False)
        sns_plot = sns.pairplot(df, hue='T-OR-D', size=1.5)
        try: os.makedirs(dest_dir)
        except: pass
        plot_name = f"scatter_plot_pca_n{n_components}_{kernel}_.png"
        sns_plot.savefig(os.path.join(dest_dir, plot_name))
        pass
    pass

def show_overall_dataset_scatter_plots(dataset, target_col=None, diag_kind=None, kind=None, corner=None, gmap_levels=None):
    dest_figures, plot_name = 'figures', 'res_scatter_plot.png'
    # res_scatter_plot = sns.pairplot(dataset, hue='T-OR-D', size=1.5)
    try: os.makedirs(dest_figures)
    except: pass
    
    try:
        if target_col is not None:
            plt.figure()
            # sns.pairplot(dataset, hue=target_col, size=1.5)
            sns.pairplot(iris, hue=target_col, palette="Set2", diag_kind="kde", height=2.5)
            # g = sns.PairGrid(dataset, hue=target_col)
            # g.map_diag(plt.hist)
            # g.map_offdiag(plt.scatter)
            # g.add_legend();
            # plt.savefig(os.path.join(dest_figures, f"plain_{plot_name}"))
    except: pass

    try:
        if diag_kind is not None:
            plt.figure()
            sns.pairplot(dataset, diag_kind=diag_kind) # # example: "kde"
            # plt.savefig(os.path.join(dest_figures, f"{diag_kind}_{plot_name}"))
    except: pass
    try:
        if kind is not None:
            plt.figure()
            sns.pairplot(dataset, kind=kind) # example: "reg"
            # plt.savefig(os.path.join(dest_figures, f"{kind}_{plot_name}"))
    except: pass
    try:
        if corner is not None:
            plt.figure()
            sns.pairplot(dataset, corner=True)
            # plt.savefig(os.path.join(dest_figures, f"corner_{plot_name}"))
    except: pass
    try:
        if gmap_levels is not None:
            plt.figure()
            g = sns.PairGrid(dataset)
            g.map_diag(sns.kdeplot)
            g.map_offdiag(sns.kdeplot, n_levels=gmap_levels) # exmaple: n_levels=6
            plt.savefig(os.path.join(dest_figures, f"gmap_{plot_name}"))
    except: pass

    pass

def show_learning_curve(dataset, plot_name, grid_size, plot_dest="figures", n=None, figsize=(5, 5), show_pairs=False, show_figure=False):

    try: os.makedirs(plot_dest)
    except: pass

    col_names = dataset.columns
    col_accs = col_names[0::2]
    col_stds = col_names[1::2]

    # print(col_names); print(col_accs); print(col_stds);

    if show_pairs is False:
        plt.figure(figsize=figsize)
    estimator_name = plot_name.split("_")[0]
    plt.title(f"Learning Curve {estimator_name}")
    # grid_shape = int(''.join([str(ii) for ii in grid_size]))
    for ii, (col_acc, col_std) in enumerate(zip(col_accs, col_stds)):

        # plt.subplot(int(f"{grid_shape}{ii+1}"))
        if show_pairs is True:
            if ii % 2 == 0:
                plt.figure()
            plt.subplot(1, 2, ii % 2 + 1)
            
            acc_list = dataset[col_acc].values[:n]
            std_list = dataset[col_std].values[:n]
        
            plt.plot(range(len(acc_list)), [float(xi) for xi in acc_list] , label='linear')

            for jj, (conf_interval, val) in enumerate(zip(std_list, acc_list)):
                conf_interval = conf_interval[-4:]
                plt.errorbar(x=jj, y=float(val), yerr=float(conf_interval), color="black", capsize=3,
                     linestyle="None",
                    marker="s", markersize=7, mfc="black", mec="black")
                pass
            if ii % 2 != 0:
                # plt.savefig(os.path.join(plot_dest, plot_name))
                plt.title(col_acc)
                plt.show()
        else:
            plt.subplot(grid_size[0], grid_size[1], ii+1)
            plt.title(col_acc)
            acc_list = dataset[col_acc].values[:n]
            std_list = dataset[col_std].values[:n]
        
            plt.plot(range(len(acc_list)), [float(xi) for xi in acc_list] , label='linear')

            for jj, (conf_interval, val) in enumerate(zip(std_list, acc_list)):
                conf_interval = conf_interval[-4:]
                plt.errorbar(x=jj, y=float(val), yerr=float(conf_interval), color="black", capsize=3,
                     linestyle="None",
                    marker="s", markersize=7, mfc="black", mec="black")
                pass
        pass

    #plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
    #                wspace=0.35)

    plt.savefig(os.path.join(plot_dest, plot_name))
    if show_pairs is True:
        if show_figure is True:
            plt.show()
    else:
        if show_figure is True:
            plt.show()
    pass

def show_learning_curve_loo_sscv(dataset, plot_name, grid_size, plot_dest="figures", n=None, col_names=None, figsize=(10, 10), show_pairs=False):
    df_res = None
    for df in dataset:
        if df_res is None: df_res = df[-2:]
        else: df_res = pd. concat([df_res, df[-2:]], ignore_index=True)
        pass
    df_res_2 = None
    estimator_name, jj = col_names[0], 0
    names_list = []
    for ii, row in enumerate(df_res.values):
        if ii % 2 == 0:
            estimator_name = col_names[jj]
            names = [estimator_name + '_loo_acc', estimator_name + '_loo_std']
        else:
            names = [estimator_name + '_Stdf_acc', estimator_name + '_Stdf_std']
            jj = jj + 1
            
        names_list.extend(copy.deepcopy(names))
        acc = pd.Series(row[0::2]) 
        std = pd.Series(row[1::2]) 
        tmp_df = pd.concat([acc, std], axis=1, ignore_index=True)
        # tmp_df = tmp_df.rename(columns=dict(zip(list(tmp_df.columns), names)))
        if df_res_2 is None: df_res_2 = tmp_df
        else: df_res_2 = pd.concat([df_res_2, tmp_df], axis=1, ignore_index=True)
        pass
    
    # col_tmp = df_res_2.columns.values.tolist()
    # return col_tmp
    # tmp_list = list(itertools.chain.from_iterable(list(zip(col_names, col_names))))
    # tmp_list = list(itertools.chain.from_iterable(list(zip(tmp_list, tmp_list))))
    # tmp_list= [f"{col_tmp[ii]}_{xi}_acc" if ii % 2 == 0 else f"{col_tmp[ii+1]}_{xi}_std" for ii, xi in enumerate(tmp_list)]
    
    df_res_2 = pd.DataFrame(df_res_2.values, columns=names_list) # df_res_2.rename(columns=dict(zip(df_res_2.columns, tmp_list)))
    plot_name = 'loo_stdf_learning_curve.png'
    show_learning_curve(df_res_2, n=df_res_2.shape[0], plot_dest=plot_dest, grid_size=[12, 2], plot_name=plot_name, figsize=figsize, show_pairs=show_pairs)
    pass
