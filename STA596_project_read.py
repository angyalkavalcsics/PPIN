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


# NOTE: change these to work for your file paths
path = 'D:/classes/STA-596/project/data/'
in_df = pd.read_pickle(f'{path}species_data.output.pickle')


# extract and merge data
sub_key = 'Taxonomy Level 2'
sub_value = 'Bacteria_Proteobacteria'
sub_df = in_df[in_df[sub_key] == sub_value]
merged = pd.merge(
    pd.read_pickle(f'{path}{sub_value}.pickle').fillna(0),
    in_df[in_df[sub_key] == sub_value].set_index('Species_ID')['Evolution'],
    on='Species_ID')
X = merged.drop(columns=['Evolution','Species_ID'])
y = merged['Evolution']


# ---- pair plot
compare = [
    'modularity',
    'mc_mean',
    'giant_prop',
    ]
sns.pairplot(X[compare], kind="reg", diag_kind="kde")


# ---- SLR
sparse_col = re.compile('(mc|d)_\d+$')
for col in X.columns[1:]:
    if sparse_col.match(col): continue
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
np.mean((y - yhat)**2) # 0.007

feature_importances = np.mean([
    tree.feature_importances_ for tree in m.estimators_
], axis=0)

for i in np.argsort(feature_importances)[::-1]:
    if feature_importances[i] < .02:
        break
    print(f'\t{X.columns[i]}: {feature_importances[i]:.3f}')