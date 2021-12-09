# -*- coding: utf-8 -*-
"""
@author: jhautala
"""

import sys
import random
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt


# ---- config
# NOTE: change to match project location
project_path = 'D:/classes/STA-596/project/'
data_path = f'{project_path}data/'
data_path = 'C:/Users/angya/OneDrive/Documents/'
tmp_path = f'{project_path}tmp/'


# ---- functions
def load_orig():
    ''' NOTE: Original pickle file must be in 'data' subdirectoryof project.
        I didn't find a better way of dodging adding it to the repo.
    '''
    df = pd.read_pickle(f'{data_path}species_data.output.pickle')
    return df.set_index('Species_ID')

def load_data():
    ''' This function loads and merges the response (for all species) with
        the predictors (see caveat), joined on 'Species_ID' as index.
        
        Caveat: this data was generated with incorrect calc for triangles.
    '''
    X = pd.read_pickle(f'{data_path}predictors.pickle')
    y = pd.read_pickle(f'{data_path}response.pickle')
    return X, y, pd.merge(X, y, on='Species_ID')

def load_subset():
    '''
        Caveat: this data was generated with incorrect calc for triangles.
    '''
    merged = pd.read_pickle(f'{data_path}Taxonomy Level 2^Bacteria_Proteobacteria.pickle')
    X = merged.drop(columns='Evolution')
    y = merged['Evolution']
    return X, y, merged

def get_extreme(df,col):
    tmp_min = df.loc[df[col] == df[col].min()].index[0]
    tmp_max = df.loc[df[col] == df[col].max()].index[0]
    return tmp_min, tmp_max

# ---- execution

src_df = load_orig()
X, y, merged = load_data()
X, y, merged = load_subset()


# ---- pair plot
compare = [
    'Evolution',
    'modularity',
    'largest_clique',
    # 'proportion_in_giant',
    'critical_threshold',
    ]
sns.pairplot(merged[compare], kind="reg", diag_kind="kde")
