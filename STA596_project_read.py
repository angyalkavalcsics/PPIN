# -*- coding: utf-8 -*-
"""
@author: jhautala
"""

import os
import re
import pandas as pd
import pickle
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn as sk
from sklearn import linear_model
from sklearn import ensemble
from sklearn import decomposition


# NOTE: You probably have to this to match your project location.
data_path = 'D:/classes/STA-596/project/data/'


# NOTE: If you want to load the original pickle file, you can just put it
#       into the data subdirectory and uncomment this.
# src_path = f'{data_path}/species_data.output.pickle'
# src_df = pd.read_pickle(src_path)


''' This function extracts and merges the predictors and response to 
align their indices and returns them separately '''
def get_merged(
        selected_sub_value=None,
        sub_key = 'Taxonomy Level 2',
        ):
    y = pd.read_pickle(f'{data_path}response.pickle')
    sub_values = ['Bacteria_Proteobacteria', 'Bacteria_Terrabacteria group',
           'Eukaryota_Stramenopiles', 'Eukaryota_Viridiplantae',
           'Eukaryota_Opisthokonta', 'Eukaryota_Euglenozoa',
           'Eukaryota_Heterolobosea', 'Eukaryota_Amoebozoa',
           'Eukaryota_Alveolata', 'Eukaryota_Rhodophyta',
           'Bacteria_FCB group', 'Archaea_Euryarchaeota',
           'Bacteria_PVC group', 'Archaea_TACK group', 'Eukaryota_Fornicata',
           'Bacteria_Fusobacteria', 'Bacteria_Acidobacteria',
           'Bacteria_Aquificae', 'Archaea_DPANN group',
           'Bacteria_Thermotogae', 'Bacteria_Spirochaetes',
           'Bacteria_Nitrospirae', 'Bacteria_Dictyoglomi',
           'Eukaryota_Parabasalia', 'Bacteria_Elusimicrobia',
           'Bacteria_Synergistetes', 'Bacteria_Deferribacteres',
           'Bacteria_Chrysiogenetes', 'Bacteria_Thermodesulfobacteria',
           'Archaea_unclassified Archaea', 'Bacteria_unclassified Bacteria']
    X = pd.DataFrame()
    for sub_value in sub_values:
        if selected_sub_value and selected_sub_value != sub_value: continue
        sub_X_path = f'{data_path}/{sub_key}^{sub_value}.pickle'
        sub_X = pd.read_pickle(sub_X_path)
        X = pd.concat([X,sub_X])
    X = X.set_index('Species_ID').fillna(0)
    pickle.dump(X, open(f'{data_path}predictors.pickle', 'wb'))
    merged = pd.merge(X, y, on='Species_ID')
    X = merged.drop(columns=['Evolution'])
    y = merged['Evolution']
    return X, y


# extract and merge data
X, y = get_merged()


# NOTE: pass 'Taxonomy Level 2' values to try different subsets
# X, y = get_merged('Bacteria_Proteobacteria')
# X, y = get_merged('Eukaryota_Opisthokonta')


# identify sparse columns
# NOTE: these features are poorly modeled as a DataFrame
sparse_cols = []
sparse_col_pattern = re.compile('(mc|d)_\d+$')
for col in X.columns[1:]:
    if sparse_col_pattern.match(col):
        sparse_cols.append(col)


# ---- pair plot
compare = [
    'modularity',
    'mc_mean',
    'giant_prop',
    ]
sns.pairplot(X[compare], kind="reg", diag_kind="kde")


# ---- SLR
sparse_cols = []
sparse_col_pattern = re.compile('(mc|d)_\d+$')
for col in X.columns[1:]:
    if sparse_col_pattern.match(col):
        sparse_cols.append(col)
        continue
    print('\n')
    model = sm.OLS(y, X[col])
    est = model.fit()
    print(est.summary())


# ---- lasso CV
lassocv = linear_model.LassoCV(eps=.001,n_alphas=250,cv=10)
lasso_est = lassocv.fit(X,y)


# ---- bagging tree
m = ensemble.BaggingRegressor()
m.fit(X,y)

yhat = m.predict(X)
np.mean((y - yhat)**2)

feature_importances = np.mean([
    tree.feature_importances_ for tree in m.estimators_
], axis=0)

for i in np.argsort(feature_importances)[::-1]:
    if feature_importances[i] < .02:
        break
    print(f'\t{X.columns[i]}: {feature_importances[i]:.3f}')
