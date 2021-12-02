# -*- coding: utf-8 -*-
"""
@author: jhautala
"""

import pandas as pd
import seaborn as sns


# ---- config
# NOTE: This is the only thing you should change (to match project location).
data_path = 'D:/classes/STA-596/project/data/'


# ---- functions
def load_data():
    ''' This function loads and merges all the predictors and response with
    'Species_ID' as index. '''
    X = pd.read_pickle(f'{data_path}predictors.pickle')
    y = pd.read_pickle(f'{data_path}response.pickle')
    return X, y, pd.merge(X, y, on='Species_ID')


# ---- execution
X, y, merged = load_data()


# ---- pair plot
compare = [
    'modularity',
    'mc_mean',
    'giant_prop',
    ]
sns.pairplot(X[compare], kind="reg", diag_kind="kde")
