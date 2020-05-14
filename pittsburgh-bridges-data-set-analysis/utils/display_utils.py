# =========================================================================== #
# IMPORTS
# =========================================================================== #

# Standard Imports (Data Manipulation and Graphics)
# --------------------------------------------------------------------------- #
import numpy as np    # Load the Numpy library with alias 'np' 
import pandas as pd   # Load the Pandas library with alias 'pd' 

import seaborn as sns # Load the Seabonrn, graphics library with alias 'sns' 

import copy
from scipy import stats

# Matplotlib pyplot provides plotting API
import matplotlib as mpl
from matplotlib import pyplot as plt
import chart_studio.plotly.plotly as py

import pprint
from sklearn.decomposition import PCA

# =========================================================================== #
# FUNCTIONS
# =========================================================================== #

def display_heatmap(corr):
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

def show_categorical_predictor_values(df, columns_2_avoid=None):
    
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