# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 09:02:48 2021
@author: angya
"""
###############################################################################
# Packages
import pandas as pd
import pickle
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as pyplot
import networkx.algorithms.community as nx_comm
from networkx.algorithms.flow import shortest_augmenting_path
import statsmodels.api as sm
from sklearn import linear_model, ensemble
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale 
###############################################################################
# Read in file
unpickled_df = pd.read_pickle("C:/Users/angya/OneDrive/Documents/species_data.output.pickle")
###############################################################################
# Take a subset of the data 
bact_prot = unpickled_df.loc[unpickled_df["Taxonomy Level 2"] == "Bacteria_Proteobacteria"]
reduced_df = bact_prot.iloc[:100, :]
adj_train = reduced_df.iloc[:, 8]
###############################################################################
# Author: Angyalka
# Plot PPIN
n = 35 # controls which species we plot
first = adj_train.iloc[n].todense()
G = nx.convert_matrix.from_numpy_matrix(first) 
# need to fix the networks so we only take the 
# giant connected component from each one 
Gcc = max(nx.connected_components(G), key=len)
giantC = G.subgraph(Gcc) 

pyplot.figure(figsize=(10,5))
ax = pyplot.gca()
ax.set_title('PPIN for ' + reduced_df['Compact Name'].iloc[n])
kawai = nx.kamada_kawai_layout(giantC)
nx.draw(giantC, kawai, node_size = 10)

###############################################################################
# Cliques
# Author: Jesse
class HistStats: 
    def __init__(self, df, val):
        self.count = df['count'].sum()
        self.max = df[val].max()
        self.mode = df.iloc[df['count'].idxmax()][val]
        self.mean = (df[val] * df['count']).sum() / self.count
        
def get_clique_count_df(G):
    clique_counts = {}
    for clique in nx.find_cliques(G):
        clique_size = len(clique)
        clique_counts[clique_size] = clique_counts.get(clique_size, 0) + 1
    
    # convert to pandas series
    clique_count_df = pd.DataFrame.from_dict(clique_counts,orient='index').reset_index()
    clique_count_df.columns = ['clique_size', 'count']
    return clique_count_df

# Degree
def get_degree_hist(G): 
    degree = nx.degree_histogram(G)
    degree_df = pd.DataFrame(degree).reset_index()
    degree_df.columns = ['degree', 'count']
    return degree_df
###############################################################################
# Run this for subset of species and store values
def get_df1(adj):
    predictors = []
    for i in range(len(adj)):
        #######################################################################
        # Author: Angyalka
        temp = adj.iloc[i].todense()
        G = nx.convert_matrix.from_numpy_matrix(temp) 
        Gcc = max(nx.connected_components(G), key=len)
        giantC = G.subgraph(Gcc) 
        centrality = nx.degree_centrality(giantC)
        centrality_df = pd.DataFrame.from_dict(centrality, orient='index')
        avg_centrality = np.mean(centrality_df.iloc[:, 0])
        # fixed 11/26
        num_triangles = int(sum(nx.triangles(giantC).values()) / 3)
        modularity = nx_comm.modularity(giantC, 
                                        nx_comm.label_propagation_communities(giantC))
        #######################################################################
        # Author: Jesse
        clique_size = HistStats(get_clique_count_df(G), 'clique_size')
        lcsg_clique_size = HistStats(get_clique_count_df(giantC), 'clique_size')
        # same as stars (repeat predictors) 
        #degree_stats = HistStats(get_degree_hist(G), 'degree')
        lcsg_degree_stats = HistStats(get_degree_hist(giantC), 'degree')
        #######################################################################
        predictors.append([
            avg_centrality,
            num_triangles,
            modularity,
            clique_size.count,
            clique_size.max,
            clique_size.mode,
            clique_size.mean,
            lcsg_clique_size.count,
            lcsg_clique_size.max,
            lcsg_clique_size.mode,
            lcsg_clique_size.mean,
            lcsg_degree_stats.count,
            lcsg_degree_stats.max,
            lcsg_degree_stats.mode,
            lcsg_degree_stats.mean,
            ])
    return(predictors)

###############################################################################
# Stars
# Author: Angyalka
'''
A k-star is a node with k edges
'''
def getstars(adj): 
        temp = adj.todense()
        G = nx.convert_matrix.from_numpy_matrix(temp) 
        Gcc = max(nx.connected_components(G), key=len)
        giantC = G.subgraph(Gcc) 
        
        A = nx.adjacency_matrix(giantC).todense()
        deg = np.asarray(np.sum(A,axis=0)).flatten()
        values, counts = np.unique(deg, return_counts=True)
        stars_sm = pd.DataFrame(counts)
        stars_sm = stars_sm.set_index(values)
        return(stars_sm)
    
def get_df2(adj):
    df_list = []
    for a in range(len(adj)):
        t = getstars(adj.iloc[a])
        df_list.append(t)
        
    stars = pd.concat(df_list, axis = 1)
    stars = stars.fillna(0)
    new_row_names = ['num_' + str(i) + 'stars' for i in stars.index ]
    stars.index = new_row_names
    return(stars.T)

###############################################################################
# Combine data frames of the various stats
# Author: Angyalka
predictors = get_df1(adj_train) 
df1 = pd.DataFrame(predictors, index = reduced_df['Species_ID'], columns=[
    'Average Centrality',
    'Number of Triangles',
    'Modularity',
    'Clique Count',
    'Clique-Size Max',
    'Clique-Size Mode',
    'Clique-Size Mean',
    'LCSG Clique Count',
    'LCSG Clique-Size Max',
    'LCSG Clique-Size Mode',
    'LCSG Clique-Size Mean',
    'LCSG Node Count',
    'LCSG Degree Max',
    'LCSG Degree Mode',
    'LCSG Degree Mean',
    ])

df2 = get_df2(adj_train)
df2.index = reduced_df['Species_ID']

X = pd.concat([df1, df2], axis=1)

y = reduced_df['Evolution']
###############################################################################
# Author: Jesse

'''
Trying some meta-stats, it seems like 'GiantProportion' (the proportion of
total nodes that are included in the largest connected subgraph) is almost
as powerful as modularity (as meaured by SLR R^2). We might expect this is
another way to measure connectivity/resiliency of the proteome.
'''

X['GiantProportion'] = X['LCSG Node Count']/X['num_1stars']

###############################################################################
# Fit LASSO and perform cv to find alpha
# Author: Angyalka

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

alp = 10**np.linspace(1,-3,100)*0.5
np.min(alp)
np.max(alp)
lasso = Lasso(max_iter = 10000, normalize = True)
coefs = []

for a in alp:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train), y_train)
    coefs.append(lasso.coef_)
    
ax = pyplot.gca()
ax.plot(alp*2, coefs)
ax.set_xscale('log')
pyplot.axis('tight')
pyplot.xlabel('alpha')
pyplot.ylabel('weights')
ax.set_title('LASSO Coefficients as a Function of Regularization', fontsize=16)

lassocv = LassoCV(alphas = None, cv = 10, max_iter = 1000000, normalize = True)
lassocv.fit(X_train, y_train)
lassocv.alphas_
lassocv.alpha_  # .0535
lasso.set_params(alpha=lassocv.alpha_)

lasso.fit(X_train, y_train)
train_pred_y = lasso.predict(X_train) 
test_pred_y = lasso.predict(X_test) 
    
print(f"train_MSE = {mean_squared_error(y_train, train_pred_y)}") # 0.0487
print(f"test_MSE = {mean_squared_error(y_test, test_pred_y)}") # 0.054

c = pd.DataFrame(lasso.coef_, index = X.columns, columns= ['coefs'])
lasso_feature = c.index[np.nonzero(np.array(c))[0]]
lasso_coefs = c.iloc[np.nonzero(np.array(c))[0]]
lasso_feature

# This returned some strange results
'''
Index(['Average Centrality', 'num_56stars', 'num_59stars', 'num_60stars',
       'num_79stars'],
      dtype='object')
'''
# LASSO feature importance plot

fig, ax = pyplot.subplots(figsize=(12, 7.5))
pyplot.rcParams['font.size'] = '12'
ax.barh(list(lasso_feature), list(lasso_coefs.iloc[:,0]))
pyplot.xlabel('Coefficient')
pyplot.ylabel('Feature')
ax.set_title('LASSO Feature Importance', fontsize=16)
pyplot.show()
###############################################################################
# Fit and tune a random forest regressor tree
# Author: Angyalka
bag_forest_reg = ensemble.RandomForestRegressor(criterion="mse",
                                       n_jobs=-1,
                                       random_state=1)

bag_forest_reg.fit(X_train, y_train)

train_pred_y = bag_forest_reg.predict(X_train)
test_pred_y = bag_forest_reg.predict(X_test)

print(f"train_MSE = {mean_squared_error(y_train, train_pred_y)}") # 0.0055
print(f"test_MSE = {mean_squared_error(y_test, test_pred_y)}") # 0.0288

param_grid = {
    "n_estimators":[100,200,300],
    "max_depth":[10, 20, 30, 40, 50],
    "max_features":[6,8,10,12,14,16, 18, 20, 25, 30]
}

rf_reg = ensemble.RandomForestRegressor()

rf_reg_tuned = GridSearchCV(estimator=rf_reg,
                            param_grid=param_grid,
                            cv=10,
                            n_jobs=-1,
                            verbose=2)

rf_reg_tuned.fit(X_train, y_train)
rf_reg_best = rf_reg_tuned.best_estimator_

train_pred_y = rf_reg_best.predict(X_train)
test_pred_y = rf_reg_best.predict(X_test)

print(f"train_MSE = {mean_squared_error(y_train, train_pred_y)}") # 0.00527
print(f"test_MSE = {mean_squared_error(y_test, test_pred_y)}") # 0.031

# Find significant predictors
feature_importances = np.mean([
    tree.feature_importances_ for tree in rf_reg_best.estimators_
], axis=0)

reduced_feat_importance = []
feature_names = []
for i in np.argsort(feature_importances)[::-1]:
    if feature_importances[i] < 0.01:
        break
    feature_names.append(X.columns[i])
    reduced_feat_importance.append(feature_importances[i])
    print(f'\t{X.columns[i]}: {feature_importances[i]:.3f}')
    
'''	
	Modularity: 0.057
	Clique Count: 0.044
	num_3stars: 0.043
	LCSG Clique-Size Mean: 0.038
	num_6stars: 0.038
	LCSG Clique Count: 0.035
	Average Centrality: 0.033
	num_1stars: 0.029
	Clique-Size Mean: 0.027
	Number of Triangles: 0.027
	num_5stars: 0.027
	num_2stars: 0.026
	LCSG Node Count: 0.026
	LCSG Clique-Size Max: 0.024
	LCSG Degree Mean: 0.024
	num_21stars: 0.020
	Clique-Size Max: 0.020
	num_13stars: 0.019
	num_12stars: 0.019
	num_7stars: 0.018
	num_15stars: 0.017
	num_17stars: 0.017
	num_11stars: 0.016
	num_9stars: 0.016
	num_27stars: 0.015
	num_25stars: 0.015
	num_10stars: 0.015
	num_8stars: 0.014
	num_4stars: 0.013
	num_16stars: 0.013
	num_22stars: 0.012
	num_52stars: 0.012
	LCSG Degree Max: 0.011
	num_14stars: 0.011
	LCSG Degree Mode: 0.011
	num_20stars: 0.011
	num_31stars: 0.011
'''
# Feature Importance plot
fig, ax = pyplot.subplots(figsize=(12, 7.5))
pyplot.rcParams['font.size'] = '12'
ax.barh(feature_names, reduced_feat_importance)
pyplot.xlabel('Importance')
pyplot.ylabel('Feature')
ax.set_title('Random Forest Feature Importance', fontsize=16)
pyplot.show()
###############################################################################
# Pairs plot to show relationship between the common significant 
# predictors from both models
# Author: Jesse
compare = {
    'Modularity',
    'num_1stars',
    'Node Count',
    'Clique Count',
    }

pp = sns.pairplot(X[compare], kind="reg", diag_kind="kde")
pp.fig.suptitle("Pairwise Relationships", y=1.00)
###############################################################################
# df visualizations 
# Author: Angyalka
df1_vis = np.round(df1.head().T, 3)
print(df1_vis.to_latex(index=True))  

df2_vis = np.round(df2.head().T.iloc[:6,:], 3)
print(df2_vis.to_latex(index=True)) 
