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
from IPython import display
import ipywidgets as widgets
from itertools import islice
import itertools

# Matplotlib pyplot provides plotting API
import matplotlib as mpl
from matplotlib import pyplot as plt
import chart_studio.plotly.plotly as py
import plotly.express as px

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

# --------------------------------------------------------------------------- #
# Descriptive Statistics Section
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# show_frequency_distribution_predictor
# --------------------------------------------------------------------------- #

def show_frequency_distribution_predictor(df, predictor_name=None, columns_2_avoid=None, features_vs_values=None, target_col=None, grid_display=False, hue=None, verbose=0):
    
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
            l = [None] * len(predictor_count.index)
            print(features_vs_values[predictor])
            revers_dict = dict()
            for k, v in features_vs_values[predictor].items():
                revers_dict[v] = k
                for ii, val in enumerate(predictor_count.index):
                    if val == v:
                        l[ii] = k
                        break
            if grid_display is True: pass
            else:
                # f = plt.figure(figsize=(10,3))
                print(predictor_count)
                print(l)
                if hue is not None:
                    _, axs = plt.subplots(1,3, figsize=(15,3))
                    sns.barplot(l, predictor_count.values, alpha=0.9, ax=axs[0])
                    axs[0].set_title('Frequency Distribution of %s' % (predictor))
                    axs[0].set_ylabel('Number of Occurrences', fontsize=12)
                    axs[0].set_xlabel('%s' % (predictor), fontsize=12)
                    df_1 = plot_hue_hist_v2(hue, predictor, features_vs_values, df, ax=axs[1], verbose=verbose)
                    df_2 = plot_hue_hist_v2(predictor, hue, features_vs_values, df, ax=axs[2], verbose=verbose)
                    if verbose == 1:
                        res = create_widget_list_obj([df_1, df_2])
                        display.display(res)
                    pass
                else:
                    sns.barplot(l, predictor_count.values, alpha=0.9)
                    plt.title('Frequency Distribution of %s' % (predictor))
                    plt.ylabel('Number of Occurrences', fontsize=12)
                    plt.xlabel('%s' % (predictor), fontsize=12)
                pass
            pass
        else:
            if hue is not None:
                sns.barplot(predictor_count.index, predictor_count.values, alpha=0.9)
                df.pivot(columns=hue)[predictor].plot(kind = 'hist', stacked=True)
            else:
                sns.barplot(predictor_count.index, predictor_count.values, alpha=0.9)
                plt.title('Frequency Distribution of %s' % (predictor))
                plt.ylabel('Number of Occurrences', fontsize=12)
                plt.xlabel('%s' % (predictor), fontsize=12)
                pass
    plt.show()
    pass

def plot_hue_hist_v2(hue, predictor, features_vs_values, df, verbose=0, ax=None):
    
    revers_dict_hue= dict()
    for k, v in features_vs_values[hue].items():
        revers_dict_hue[v] = k
    revers_dict = dict()
    for k, v in features_vs_values[predictor].items():
        revers_dict[v] = k
    
    res = df.groupby(predictor)[hue].value_counts()
    tmp_res = res.unstack(0).values
    tmp_index = list(map(lambda xi: revers_dict_hue[xi], res.unstack(0).index.values))
    tmp_col = list(map(lambda xi: revers_dict[xi], res.unstack(0).columns))

    df_tmp = pd.DataFrame(tmp_res, columns=tmp_col, index=tmp_index).head()
    if verbose == 1:
        print(df_tmp.head())
    df_tmp.plot.bar(stacked=True, ax=ax)
    ax.set_title('Frequency Distribution of %s over %s' % (predictor, hue))
    ax.set_ylabel('Number of Occurrences', fontsize=12)
    ax.set_xlabel('%s' % (hue,), fontsize=12)
    return df_tmp

def create_widget_list_obj(list_objs):
    res_list = []
    for item in list_objs:
        widget = widgets.Output()
        with widget: display.display(item); pass
        res_list.append(widget)
        pass
    hbox = widgets.HBox(res_list)
    return hbox


def plot_hue_hist(hue, predictor, predictor_count, features_vs_values, df, revers_dict):
    revers_dict_hue= dict()
    for k, v in features_vs_values[hue].items():
        revers_dict_hue[v] = k
        # df.pivot(columns=hue)[predictor].plot(kind = 'hist', stacked=True)
    data = dict()
    for pos, idx in enumerate(predictor_count.index):
        val = df.groupby(predictor).get_group(idx)[hue]
        # data_tmp = map(lambda xi: revers_dict_hue[xi], val.values)
        # val_tmp = pd.Series(data=data_tmp, index=val.index)
        key = revers_dict[idx]
        data[key] = val #_tmp
        pass
    print(pd.DataFrame(data).head())
    pd.DataFrame(data).plot(kind = 'hist', stacked=True,)
    return
    # pd.DataFrame(data).size().unstack().plot(kind='bar', stacked=True, figsize=(15, 5))
    plt.title('Frequency Distribution of %s over %s' % (predictor, hue))
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('%s' % (hue), fontsize=12)
    plt.show()
    data = dict()
    for pos, idx in enumerate(df[hue].value_counts().index):
        val = df.groupby(hue).get_group(idx)[predictor]
        key = revers_dict_hue[idx]
        data[key] = val
        pass
    pd.DataFrame(data).plot(kind = 'hist', stacked=True,)
    # pd.DataFrame(data).size().unstack().plot(kind='bar', stacked=True, figsize=(15, 5))
    plt.title('Frequency Distribution of %s over %s' % (hue, predictor))
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('%s' % (predictor), fontsize=12)
    plt.show()
    pass

# --------------------------------------------------------------------------- #
# Others
# --------------------------------------------------------------------------- #

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
        # pprint(predictor)
        # pprint(labels)
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

# --------------------------------------------------------------------------- #
# CV Results Plots
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
# Pie and Hist Section
# --------------------------------------------------------------------------- #

def show_pie_charts_corr_matrix(corr_matrix, subplots=False):
    
    err_msg_assert = f"Error: input correlation matrix is not of type pd.DataFrame, but acctually is of type {type(corr_matrix)}"
    assert type(corr_matrix) is pd.DataFrame, err_msg_assert

    colors=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:purple"]

    if subplots is True:
        fig_, axs = plt.subplots(4)
    data_norm, data = prepare_data_corr_matrix_pie(corr_matrix)
    if subplots is True:
        fig = show_pie_chart_corr_matrix(data, colors[:3], subplots, axs[0])
    else:
        fig = show_pie_chart_corr_matrix(data, colors[:3])

    data, data_2 = prepare_data_corr_matrix_hist(corr_matrix)
    if subplots is True:
        fig = show_stack_histogram_corr_matrix(data, data_2, corr_matrix, colors[:3], subplots, axs[1])
    else:
        fig = show_stack_histogram_corr_matrix(data, data_2, corr_matrix, colors[:3])
    
    data_norm, data = prepare_data_corr_matrix_pie_finer_analysis(corr_matrix)
    if subplots is True:
        fig = show_pie_chart_corr_matrix_finer_analysis(data, colors, subplots, axs[2])
    else:
        fig = show_pie_chart_corr_matrix_finer_analysis(data, colors)

    colors=["blue", "orange", "green", "red", "purple", "brown"]
    data, data_2 = prepare_data_corr_matrix_hist_v2(corr_matrix)
    if subplots is True:
        fig = show_stack_histogram_corr_matrix(data, data_2, corr_matrix, colors, subplots, axs[3])
    else:
        fig = show_stack_histogram_corr_matrix(data, data_2, corr_matrix, colors)

    # plt.tight_layout()
    # plt.show()
    pass


def show_stack_histogram_corr_matrix(data, data_2, corr_matrix, colors, title="a histogram for corr matrix", subplots=False, ax=None):
    # index_tmp = ["weak", "moderate", "strong"]
    # df = pd.DataFrame(data=data, columns=index_tmp, index=corr_matrix.columns)
    # plt.figure()

    # dict_vals = dict(corr_matrix.columns, zip(data_2))
    # df = pd.DataFrame(data=dict_vals)
    df = pd.DataFrame(data=data_2, columns=["Attribute", "Type Corr"])
    # f.groupby(["Attribute", "Type Corr"]).size().unstack().plot(kind='bar',stacked=True, colors=colors)
    if subplots is True:
        fig = df.groupby(["Attribute", "Type Corr"]).size().unstack().plot(kind='bar', stacked=True, layout=(2,2), subplots=subplots, ax=ax)
    else:
        fig = df.groupby(["Attribute", "Type Corr"]).size().unstack().plot(kind='bar', stacked=True)
    # plt.show()
    return fig


def show_pie_chart_corr_matrix(data, colors, title="a pie chart for corr matrix", subplots=False, ax=None):
    index_tmp = ["moderate", "weak", "strong"]

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct

    index = []
    for d, i in zip(data, index_tmp):
        index.append(f"{i}({d*100:.2f}%)" )
    df = pd.DataFrame(data=data, columns=["Correlation"], index=index)
    if subplots is True:
        fig = df.plot.pie(y='Correlation', title=title, figsize=(6, 6), layout=(2,2), autopct=make_autopct(df["Correlation"].values), subplots=subplots, ax=ax)
    else:
        # fig = df.plot.pie(y='Correlation', labels=index_tmp, figsize=(5, 5), autopct="%.2f%%",)
        fig = df.plot.pie(y='Correlation', labels=index_tmp, figsize=(6, 6), autopct=make_autopct(df["Correlation"].values), colors=colors)
    return fig


def show_pie_chart_corr_matrix_finer_analysis(data, colors, title="a pie chart for corr matrix", subplots=False, ax=None):
    """"https://plotly.com/python/pie-charts/"""
    index_tmp = ["moderate", "weak", "strong"]

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct

    colors=["tab:blue", "tab:orange", "tab:brown", "tab:green", "tab:red", "tab:purple"]

    index = []
    labels = []
    tot = sum(data)
    for i, ii in enumerate(range(0, len(data)//2)):
        d = data[ii]
        i = f"Neg - {index_tmp[i]}"
        index.append(f"{i}({d * 100 / tot:.2f}%)" )
        labels.extend([i])
        pass
    for i, ii in enumerate(range(len(data)//2, len(data))):
        d = data[ii]
        i = f"Pos - {index_tmp[i]}"
        index.append(f"{i}({d * 100 / tot:.2f}%)" )
        labels.extend([i])
        pass
    df = pd.DataFrame(data=data, columns=["Correlation"], index=index)
    if subplots is True:
        fig = df.plot.pie(y='Correlation', title=title, figsize=(6, 6), layout=(2,2), autopct=make_autopct(df["Correlation"].values), subplots=subplots, ax=ax)
    else:
        # fig = df.plot.pie(y='Correlation', figsize=(5, 5), autopct='%1.1f%%',)
        fig = df.plot.pie(y='Correlation', labels=labels, title=title, figsize=(8, 8), autopct=make_autopct(df["Correlation"].values), colors=colors)
    # fig = px.pie(df, values='pop', names='country', title='Population of European continent')
    # fig.show()
    return fig


def prepare_data_corr_matrix_pie(corr_matrix):
    
    err_msg_assert = f"Error: input correlation matrix is not of type pd.DataFrame, but acctually is of type {type(corr_matrix)}"
    assert type(corr_matrix) is pd.DataFrame, err_msg_assert


    moderate = lambda xi: .5 < xi < .8
    weak = lambda xi: .5 >= xi
    strong = lambda xi: xi >= .8
    
    cnts = [0] * 3
    lambdas = [moderate, weak, strong]
    data_2 = []
    for ii, row in enumerate(corr_matrix.values):
        if ii == 0:
            filter_indices = range(1, len(row))
        elif ii == len(row) - 1:
            # filter_indices = range(0, len(row)-1)
            break
        else:
            # filter_indices = list(range(0, ii-1)) + list(range(ii+1, len(row)))
            filter_indices = list(range(ii+1, len(row)))
        
        tmp_row = np.take(np.absolute(row), filter_indices)
        record = []
        for ii, lamnda_func in enumerate(lambdas):
            var = len(list(filter(lamnda_func, tmp_row)))
            record.append(var)
            cnts[ii] = cnts[ii] + var
            pass
        data_2.append(record)
        pass
    from sklearn.preprocessing import Normalizer
    data = np.array(cnts, ndmin=2)
    
    data_norm = Normalizer().fit_transform(X=data[:]).flatten()
    return data_norm, data.flatten()


def prepare_data_corr_matrix_hist(corr_matrix):
    
    index_tmp = ["weak", "moderate", "strong"]

    err_msg_assert = f"Error: input correlation matrix is not of type pd.DataFrame, but acctually is of type {type(corr_matrix)}"
    assert type(corr_matrix) is pd.DataFrame, err_msg_assert

    weak = lambda xi: .5 >= xi
    moderate = lambda xi: .5 < xi < .8
    strong = lambda xi: xi >= .8
    
    cnts = [0] * 3
    lambdas = [weak, moderate, strong]
    data, data_2 = [], []
    for ii, row in enumerate(corr_matrix.values):
        if ii == 0:
            filter_indices = range(1, len(row))
        elif ii == len(row) - 1:
            filter_indices = range(0, len(row)-1)
            break
        else:
            filter_indices = list(range(0, ii-1)) + list(range(ii+1, len(row)))
        
        tmp_row = np.take(np.absolute(row), filter_indices)
        record = []
        record_2 = []
        for jj, lamnda_func in enumerate(lambdas):
            var = len(list(filter(lamnda_func, tmp_row)))
            record.append(var)
            cnts[jj] = cnts[jj] + var

            if var == 0: continue

            vals = [index_tmp[jj]] * var
            attr = [list(corr_matrix.columns)[ii]] * var
            tmp_val = list(map(lambda xi: [xi[0], xi[1]], zip(attr, vals)))
            record_2.extend(tmp_val)

            pass
        # print(record)
        data.append(record)
        data_2.extend(record_2)
        pass
    return data, data_2


def prepare_data_corr_matrix_hist_v2(corr_matrix):
    
    index_tmp = ["moderate", "strong", "weak"]

    err_msg_assert = f"Error: input correlation matrix is not of type pd.DataFrame, but acctually is of type {type(corr_matrix)}"
    assert type(corr_matrix) is pd.DataFrame, err_msg_assert
    
    index = []
    for i, ii in enumerate(range(0, 3)):
        id = f"Neg - {index_tmp[i]}"
        index.extend([id])
    for i, ii in enumerate(range(0, 3)):
        id = f"Pos - {index_tmp[i]}"
        index.extend([id])

    neg_moderate = lambda xi: -.5 > xi > -.8
    neg_strong = lambda xi: xi <= -.8
    neg_weak = lambda xi: -.5 <= xi < 0
    
    
    pos_moderate = lambda xi: .5 < xi < .8
    pos_strong = lambda xi: xi >= .8
    pos_weak = lambda xi: .5 >= xi >= 0
    

    lambdas = [neg_moderate, neg_strong, neg_weak, pos_moderate, pos_strong, pos_weak]
    
    cnts = [0] * len(lambdas)
    data, data_2 = [], []
    for ii, row in enumerate(corr_matrix.values):
        if ii == 0:
            filter_indices = range(1, len(row))
        elif ii == len(row) - 1:
            filter_indices = range(0, len(row)-1)
            break
        else:
            filter_indices = list(range(0, ii-1)) + list(range(ii+1, len(row)))
        
        tmp_row = np.take(row, filter_indices)
        record = []
        record_2 = []
        for jj, lamnda_func in enumerate(lambdas):
            var = len(list(filter(lamnda_func, tmp_row)))
            record.append(var)
            cnts[jj] = cnts[jj] + var
            if var == 0: continue

            vals = [index[jj]] * var
            attr = [list(corr_matrix.columns)[ii]] * var
            tmp_val = list(map(lambda xi: [xi[0], xi[1]], zip(attr, vals)))
            
            record_2.extend(tmp_val)
            pass
        # print(record)
        data.append(record)
        data_2.extend(record_2)
        pass
    return data, data_2


def prepare_data_corr_matrix_pie_finer_analysis(corr_matrix):
    
    err_msg_assert = f"Error: input correlation matrix is not of type pd.DataFrame, but acctually is of type {type(corr_matrix)}"
    assert type(corr_matrix) is pd.DataFrame, err_msg_assert

    neg_moderate = lambda xi: -.5 > xi > -.8
    neg_strong = lambda xi: xi <= -.8
    neg_weak = lambda xi: -.5 <= xi < 0

    pos_moderate = lambda xi: .5 < xi < .8
    pos_strong = lambda xi: xi >= .8
    pos_weak = lambda xi: .5 >= xi >= 0
    

    lambdas = [neg_moderate, neg_weak, neg_strong, pos_moderate, pos_weak, pos_strong]
    
    cnts = [0] * len(lambdas)
    for ii, row in enumerate(corr_matrix.values):
        if ii == 0:
            filter_indices = range(1, len(row))
        elif ii == len(row) - 1:
            # filter_indices = range(0, len(row)-1)
            break
        else:
            # filter_indices = list(range(0, ii-1)) + list(range(ii+1, len(row)))
            filter_indices = list(range(ii+1, len(row)))
        tmp_row = np.take(row, filter_indices)
        for ii, lamnda_func in enumerate(lambdas):
            cnts[ii] = cnts[ii] + len(list(filter(lamnda_func, tmp_row)))
        pass
    from sklearn.preprocessing import Normalizer
    data = np.array(cnts, ndmin=2)
    
    data_norm = Normalizer().fit_transform(X=data[:]).flatten()
    return data_norm, data.flatten()

# --------------------------------------------------------------------------- #
# Pie Charts in Python:
# --------------------------------------------------------------------------- #
# - https://plotly.com/python/pie-charts/
# - http://queirozf.com/entries/pandas-dataframe-plot-examples-with-matplotlib-pyplot
