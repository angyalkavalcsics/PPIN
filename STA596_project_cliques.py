# -*- coding: utf-8 -*-
"""
@author: jhautala
"""

import sys
import os
import re
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn as sk
from sklearn import ensemble
from sklearn import decomposition
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import scipy as sp
from scipy.special import comb
from tqdm import tqdm
from time import sleep
from collections import Counter
import multiprocessing as mp
import queue
from memory_profiler import profile
import shutil


# NOTE: change paths as needed
project_path = 'D:/classes/STA-596/project/'
tmp_path = f'{project_path}tmp/'
data_path = f'{project_path}data/'


# import local code (to satisfy requirements of parallelism)
sys.path.append(project_path)
import defs


# ---- constants
sparse_col_pattern = re.compile('(Total |LCSG )?(mc|d)_\d+$')
tl2_pattern = re.compile('^Taxonomy Level 2\^(.+)\.pickle$')

# NOTE: If you want to load the original pickle file, you can put it into the
#       data subdirectory and uncomment this.
src_path = f'{data_path}/species_data.output.pickle'
src_df = pd.read_pickle(src_path)


# ---- functions
# identify and remove "sparse" columns
# (i.e. exhaustive counts of stars, degrees and cliques)
# TOTE: These features are poorly modeled as a DataFrame.
#       We should probably just drop this line of inquiry
#       and just use summary stats for maximal clique size
#       and degree features.
def drop_sparse_columns(df):
    sparse_cols = []
    for col in df.columns:
        if sparse_col_pattern.match(col):
            sparse_cols.append(col)
    return df.drop(columns=sparse_cols)

def load_data():
    ''' This function loads and merges all the predictors and response with
    'Species_ID' as index. '''
    X = pd.read_pickle(f'{data_path}predictors.pickle')
    y = pd.read_pickle(f'{data_path}response.pickle')
    return X, y, pd.merge(X, y, on='Species_ID')

def load_data_orig(drop_sparse=True):
    ''' This function loads and merges all the predictors and response with
    'Species_ID' as index. '''
    X = pd.DataFrame()
    y = pd.read_pickle(f'{data_path}response.pickle')
    for tmp_file in os.listdir(data_path):
        match = tl2_pattern.match(tmp_file)
        if match:
            print(f'loading ${match.group(1)}')
            sub_X = pd.read_pickle(f'{data_path}/{tmp_file}')
            X = pd.concat([X,sub_X])
    
    X = X.set_index('Species_ID').fillna(0)
    if drop_sparse:
        X = drop_sparse_columns(X)
    pickle.dump(X, open(f'{data_path}predictors.pickle', 'wb'))
    return X, y, pd.merge(X, y, on='Species_ID')

def standardize(df):
    df_std = pd.DataFrame(StandardScaler().fit_transform(df))
    
    # re-add columns and index after preprocessing
    df_std.columns = merged.columns
    df_std.index = merged.index
    X_std = df_std.drop(columns=['Evolution'])
    y_std = df_std['Evolution']
    return X_std, y_std, df_std

def calculate(
        in_df,
        sub_key,
        sub_value,
        num_items=-1,
        prior_item_count=0, # used to resume prior execution
    ):
    file_prefix = f'{tmp_path}/{sub_key}^{sub_value}'
    
    if prior_item_count == 0:
        prior_X = pd.DataFrame()
    else:
        prior_X = pd.read_pickle(f'{file_prefix}_{prior_item_count}.pickle')
    prior_item_count = prior_X.shape[0]
    
    # select subset
    sub_df = in_df.loc[in_df[sub_key] == sub_value]
    if num_items > 0:
        sub_df = sub_df.iloc[prior_item_count:num_items,:]
    else:
        sub_df = sub_df.iloc[prior_item_count:,:]

    
    num_items = sub_df.shape[0]
    out_path = f'{file_prefix}_{num_items}.pickle'
    
    cores = mp.cpu_count()
    num_workers = max(1, cores//2)
    
    progress = tqdm(total=num_items)
    df_queue = mp.Queue(num_workers*2)
    row_queue = mp.Queue(num_workers*2)
    result_queue = mp.Queue()

    # start producer
    producer = defs.Producer(sub_df, num_workers, df_queue)
    producer.start()
    
    # start workers
    workers = []
    for i in (range(num_workers)):
        worker = defs.Worker(df_queue, row_queue)
        worker.start()
        workers.append(worker)
    
    # start consumer
    consumer = defs.Consumer(
        num_workers,
        row_queue,
        result_queue,
        )
    consumer.start()
    
    # get progress updates
    while True:
        try:
            X = result_queue.get_nowait()
            if X is None:
                break
            else:
                pickle.dump(pd.concat([prior_X,X]), open(out_path, 'wb'))
                if not progress.update():
                    progress.refresh()
        except queue.Empty:
            sleep(.5)
    
    producer.join()
    for worker in workers:
        worker.join()
    consumer.join()
    progress.close()


def get_predictors(path,drop_sparse=True):
    tmp_file_pattern = re.compile('(.+)\^(.+)(_\d+)?\.pickle$')
    tmp_df = pd.DataFrame()
    for tmp_file in os.listdir(path):
        match = tmp_file_pattern.match(tmp_file)
        if match:
            file = f'{path}{tmp_file}'
            df = pd.read_pickle(file)
            tmp_df = pd.concat([tmp_df,df])
    
    cols = []
    sparse_cols = []
    sparse_col_pattern = re.compile('(Total |LCSG )?(mc|d)_\d+$')
    for col in tmp_df.columns[1:]:
        if sparse_col_pattern.match(col):
            sparse_cols.append(col)
        else:
            cols.append(col)
        
    return tmp_df.set_index('Species_ID').fillna(0)

@profile
def main():
    # src_df = pd.read_pickle(f'{data_path}species_data.output.pickle')
    
    
    # ---- spot-calculate single subset
    # calculate(src_df,'Taxonomy Level 2','Bacteria_Proteobacteria')
    
    
    # ---- iterate over subsets
    sub_key = 'Taxonomy Level 2'
    for sub_value in src_df[sub_key].unique():
        calculate(src_df,sub_key,sub_value)
    
    
    # ---- copy from tmp to data
    # tmp_file_pattern = re.compile('(.+)\^(.+)_\d+\.pickle$')
    # for tmp_file in os.listdir(tmp_path):
    #     match = tmp_file_pattern.match(tmp_file)
    #     if match:
    #         out_file = f'{data_path}{match.group(1)}^{match.group(2)}.pickle'
    #         shutil.copy(f'{tmp_path}{tmp_file}', out_file)
            
    

if __name__ == "__main__":
    mp.freeze_support()
    main()
