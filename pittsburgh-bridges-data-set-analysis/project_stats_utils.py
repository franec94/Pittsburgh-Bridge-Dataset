print(__doc__)

# =========================================================================== #
# Imports
# =========================================================================== #

# Imports through 'from' syntax
# --------------------------------------------------------------------------- #
from itertools import islice;
from pprint import pprint;
from sklearn import preprocessing;


# Standard Imports
# --------------------------------------------------------------------------- #
import copy; import os;
import sys; import shutil;
import time;


# Imports through 'as' syntax
# --------------------------------------------------------------------------- #
import numpy as np; import pandas as pd;

# Matplotlib pyplot provides plotting API
import matplotlib as mpl
from matplotlib import pyplot as plt
import chart_studio.plotly.plotly as py
import seaborn as sns; sns.set()

# =========================================================================== #
# Functions
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Fetch Input data Functions
# --------------------------------------------------------------------------- #
def dir_traversal_by_os_walk(root_dir_path: str, verbose: int = 0, filter_access_files: bool = False) -> list:
    resources_list: list = [(root, dirs, files) for root, dirs, files, in os.walk(root_dir_path)]  
    
    if verbose == 1:
        print("List of all sub-directories and files:")  
        for (root, dirs, files)  in resources_list: 
            print('Root:', root)
            print('Directories:', dirs)
            print('Files:', files)
            pass
        pass
    
    if filter_access_files is True:
        def check_permission_file(a_file) -> bool:
            return  os.access(a_file, os.F_OK) and \
                (os.access(a_file, os.R_OK) \
                or os.access(a_file, os.W_OK) \
                or os.access(a_file, os.X_OK))
        res = []
        for ii, (root, dirs, files) in enumerate(resources_list):
            files_tmp = list(filter(check_permission_file, files))
            res.append((root, dirs, files_tmp))
            pass
        resources_list = res
        pass
    return resources_list

# --------------------------------------------------------------------------- #
# Create Dataframes Functions
# --------------------------------------------------------------------------- #
def get_df_from_list_of_os_walk_numeric(resources_list: list, columns="root,dirs,files", verbose: int = 1) -> pd.DataFrame:
    if type(columns) is not list:
        columns = "root,dirs,files".split(",")
    stats_list: list = list(map(lambda record: (record[0], len(record[1]), len(record[2])), resources_list))
    df: pd.DataFrame = pd.DataFrame(data=stats_list, columns=columns)
    return df

def get_df_from_list_of_os_walk(resources_list: list, columns="root,dirs,files,files size", verbose: int = 1) -> pd.DataFrame:
    data: list = list()
    if type(columns) is not list:
        columns = columns.split(",")
    for _, (root, dirs, files) in enumerate(resources_list):
        for _, a_file in enumerate(files):
            a_record: list = [root, os.path.dirname(a_file), os.path.basename(a_file), os.path.getsize(os.path.join(root, a_file))]
            data.append(a_record)
            pass
        pass
    df: pd.DataFrame = pd.DataFrame(data=data, columns=columns)
    return df

def get_df_from_list_of_os_walk_numeric_indexed(resources_list: list, columns="dirs,files", verbose: int = 1) -> pd.DataFrame:
    if type(columns) is not list:
        columns = columns.split(",")
    stats_list: list = list(map(lambda record: (len(record[1]), len(record[2])), resources_list))
    index_list: list = list(map(lambda record: record[0], resources_list))
    
    df: pd.DataFrame = pd.DataFrame(data=stats_list, columns=columns, index=index_list)
    return df

def get_df_from_list_of_os_walk_numeric_indexed_v2(resources_list: list, columns="dirs,dirs_size,files", verbose: int = 1) -> pd.DataFrame:
    if type(columns) is not list:
        columns = columns.split(",")
    
    def get_size_all_files(root, files_list: list) -> int:
        tot_size = 0
        for _, a_file in enumerate(list(map(lambda xi: os.path.join(root, xi), files_list))):
            try:
                if os.path.exists(a_file) and os.path.isfile(a_file):
                    tot_size = tot_size + os.path.getsize(a_file)
            except: pass
        return tot_size
    
    stats_list: list = list(map(lambda record: (len(record[1]), get_size_all_files(record[0], record[2]), len(record[2])), resources_list))
    index_list: list = list(map(lambda record: record[0], resources_list))
    
    df: pd.DataFrame = pd.DataFrame(data=stats_list, columns=columns, index=index_list)
    return df

# --------------------------------------------------------------------------- #
# Show Graphs Functions
# --------------------------------------------------------------------------- #
def show_histograms_by_scaler_tech(df: pd.DataFrame, variable_name: str, rescale_data_techs: list, save_fig=False, rescale_data_tech: str=None, dest_fig: str="stats_report_figures", fig_name: str="all_hists_chart.ong", figsize=None, show_default=False):
    
    assert df is not None, "df is None"
    assert variable_name in df.columns, f"variable name: {variable_name} is not a column name of input dataframe"
    
    check_and_create_dirs(dest_fig)

    n = len(rescale_data_techs)
    nrows = n // 2 if n % 2 == 0 else n // 2 + 1
    ncols = 2
    
    if figsize is not None: plt.figure(figsize=figsize)
    else: plt.figure()
    
    if show_default is True:
        index = 1
        plt.subplot(nrows, ncols, 1)
        meta_data_img = {
                'title': f'Hist',
                'ylabel': 'Freq',
                'xlabel': 'Ext'
        }
        show_histogram_by_variable_from_df(
            df, variable_name=variable_name,
            fig_name="plain_hist.png",
            meta_data_img=meta_data_img)
    else: index = 0
    for _, scaler_tech in enumerate(rescale_data_techs):
        
        # index = 1 if ii % 2 == 0 else 2
        index = index + 1
        plt.subplot(nrows, ncols, index)
        meta_data_img = {
                'title': f'Hist|{scaler_tech.lower()}',
                'ylabel': 'Freq',
                'xlabel': 'Ext'
        }
        show_histogram_by_variable_from_df(
            df, variable_name=variable_name,
            rescale_data_tech=scaler_tech,
            fig_name="plain_hist_{}.png".format(scaler_tech.lower()),
            meta_data_img=meta_data_img)
        # if index == 2: nrows = nrows + 1
        pass
    plt.show()
    if save_fig is True:
        plt.savefig(os.path.join(dest_fig, fig_name))
        pass
    pass

def show_histogram_by_variable_from_df(df: pd.DataFrame, variable_name: str, save_fig=False, rescale_data_tech: str=None, dest_fig: str="stats_report_figures", fig_name="a_hist_chart.png", meta_data_img: dict=None) -> None:
    
    assert df is not None, "df is None"
    assert variable_name in df.columns, f"variable name: {variable_name} is not a column name of input dataframe"
    
    check_and_create_dirs(dest_fig)
    
    labels_list = "title,ylabel,xlabel".split(",")
    values_list = "Histogram,Y,X".split(",")
    
    predictor = df[f"{variable_name}"].value_counts()
    if rescale_data_tech is not None:
        tech_names: list = "MinMaxScaler,StandardScaler,normalize".lower().split(",")
        tech_list: list = [preprocessing.MinMaxScaler(),preprocessing.StandardScaler(), preprocessing.normalize]
        tech_dict: dict = dict(zip(tech_names, tech_list))
        
        rescaled_values = check_perform_rescaling_technique(predictor.values, rescale_data_tech, tech_dict)
     
        ax = sns.barplot(predictor.index, rescaled_values, alpha=0.9)
    else:
        ax = sns.barplot(predictor.index, predictor.values, alpha=0.9)
    
    if meta_data_img is not None:
        set_meta_data_img_hist(ax, meta_data_img, labels_list, values_list)
    
    if save_fig is True:
        ax.savefig(os.path.join(dest_fig, fig_name))
        pass
    pass

def show_pie_by_variable_from_df(df: pd.DataFrame, variable_name: str, save_fig=False, dest_fig: str="stats_report_figures", fig_name="a_pie_chart.png", meta_data_img: dict=None) -> None:
    assert df is not None, "df is None"
    assert variable_name in df.columns, f"variable name: {variable_name} is not a column name of input dataframe"
    
    check_and_create_dirs(dest_fig)
    
    labels_list = "title,ylabel,xlabel".split(",")
    values_list = "Pie,Y,X".split(",")

    predictor = df[f"{variable_name}"].value_counts()
    tmp_df = pd.DataFrame(data=predictor.values, columns=[f"{variable_name}"], index=predictor.index)
    tmp_df.plot.pie(y=f"{variable_name}", figsize=(5, 5))
    
    if meta_data_img is not None:
        # set_meta_data_img_hist(ax, meta_data_img, labels_list, values_list)
        pass
    
    if save_fig is True:
        ax.savefig(os.path.join(dest_fig, fig_name))
        pass
    pass

# --------------------------------------------------------------------------- #
# Utils Functions
# --------------------------------------------------------------------------- #
def check_and_create_dirs(dir_name: str) -> None:
    try: os.makedirs(dir_name)
    except: pass
    pass

def check_perform_rescaling_technique(data, technique_name: str, techniques_dict: dict) -> object:
    tmp_technique_name: str= technique_name.lower()
    if tmp_technique_name not in techniques_dict.keys():
        raise f"Error: {technique_name} is not allowed."
        
    a_scaler = techniques_dict[f"{tmp_technique_name}"]
    if tmp_technique_name != 'normalize':
        pred_rescaled = a_scaler.fit_transform(data[:,np.newaxis]).ravel()
    else:
        pred_rescaled= a_scaler(data[:,np.newaxis], axis=0).ravel()
    return pred_rescaled

def set_meta_data_img_hist(ax, meta_data_img:dict, labels_list: list, values_list: list):

    for ii, label_img in enumerate(labels_list):
        if label_img not in meta_data_img.keys():
            meta_data_img[f"{label_img}"] = value_list[ii]
        pass
    ax.set_title('{}'.format(meta_data_img["title"]))
    ax.set_ylabel('{}'.format(meta_data_img["ylabel"]))
    ax.set_xlabel('{}'.format(meta_data_img["xlabel"]))
    pass

def rescale_data_by_technique(data, technique_name: str, techniques_list) -> object:
    
        
    
    return pred_rescaled