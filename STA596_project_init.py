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
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale, StandardScaler
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
n = 0 # controls which species we plot
first = adj_train.iloc[n].todense()
G = nx.convert_matrix.from_numpy_matrix(first) 
# need to fix the networks so we only take the 
# giant connected component from each one 
Gcc = max(nx.connected_components(G), key=len)
giantC = G.subgraph(Gcc) 

kcore = nx.k_core(giantC)
K = max(nx.connected_components(kcore), key=len)
giantK = G.subgraph(K) 
pyplot.subplot(121)
pyplot.title('Full graph')
nx.draw(giantC, node_size=10)
pyplot.subplot(122)
pyplot.title('Graph core with k = 10')
nx.draw(giantK, node_size=10)
pyplot.show()

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
    find_max_clique = 0
    max_clique = []
    for clique in nx.find_cliques(G):
        clique_size = len(clique)
        if(clique_size > find_max_clique): 
            find_max_clique = clique_size
            max_clique = list(clique)
        clique_counts[clique_size] = clique_counts.get(clique_size, 0) + 1
    
    # convert to pandas series
    clique_count_df = pd.DataFrame.from_dict(clique_counts,orient='index').reset_index()
    clique_count_df.columns = ['clique_size', 'count']
    return clique_count_df, max_clique

# Degree
def get_degree_hist(G): 
    degree = nx.degree_histogram(G)
    degree_df = pd.DataFrame(degree).reset_index()
    degree_df.columns = ['degree', 'count']
    return degree_df

# visual for how large cliques support network topology
get_cliques = get_clique_count_df(giantC)
clique_size = HistStats(get_cliques[0], 'clique_size')
lcsg_nodes = get_cliques[1]
len(lcsg_nodes)
lcsg = giantC.subgraph(lcsg_nodes) 
        
pyplot.figure(figsize=(10,5))
ax = pyplot.gca()
ax.set_title('26-clique within Desulfovibrio vulgaris Hildenborough PPIN')
nx.draw(lcsg, node_size = 45)
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
        centrality = nx.degree_centrality(G)
        centrality_df = pd.DataFrame.from_dict(centrality, orient='index')
        avg_centrality = np.mean(centrality_df.iloc[:, 0])
        # fixed 11/26
        num_triangles = int(sum(nx.triangles(G).values()) / 3)
        modularity = nx_comm.modularity(G, 
                                        nx_comm.label_propagation_communities(G))
        # A complete graph has density 1; the minimal density of any graph is 0.
        density = nx.density(G)
        #######################################################################
        # Author: Jesse and Angyalka
        get_cliques = get_clique_count_df(G)
        clique_size = HistStats(get_cliques[0], 'clique_size')
        # fixed to find largest complete SG 12/14
        # finds clique of maximum size
        # a clique is a complete subgraph of a given graph
        lcsg_get_cliques = get_clique_count_df(giantC)
        lcsg_clique_size = HistStats(lcsg_get_cliques[0], 'clique_size')
        lcsg_max_clique = lcsg_get_cliques[1]
        get_clique_edges = giantC.subgraph(lcsg_max_clique)
        max_clique_edges = nx.number_of_edges(get_clique_edges)
        # same as stars (repeat predictors) 
        #degree_stats = HistStats(get_degree_hist(G), 'degree')
        lcsg_degree_stats = HistStats(get_degree_hist(giantC), 'degree')
        lcsg_density = nx.density(giantC)
        lcsg_alg_conn = nx.algebraic_connectivity(giantC)
        lcsg_modularity = nx_comm.modularity(giantC, 
                                        nx_comm.label_propagation_communities(giantC))
        #######################################################################
        predictors.append([
            avg_centrality,
            num_triangles,
            modularity,
            density, 
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
            lcsg_density,
            lcsg_alg_conn,
            lcsg_modularity,
            max_clique_edges,
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
    'Density',
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
    'LCSG Denisty',
    'LCSG Algebraic Connectivity',
    'LCSG Modularity',
    'Number Edges Main Core'
    ])

df2 = get_df2(adj_train)
df2.index = reduced_df['Species_ID']

###############################################################################
# Get additional predictors
# Author: Jesse

data_path = 'C:/Users/angya/OneDrive/Documents/'

# ---- functions
def load_data():
    ''' This function loads and merges all the predictors and response with
    'Species_ID' as index. '''
    X = pd.read_pickle(f'{data_path}predictors.pickle')
    y = pd.read_pickle(f'{data_path}response.pickle')
    return X, y, pd.merge(X, y, on='Species_ID')


# ---- execution
X_pickle, y_pickle, merged = load_data()

#np.allclose(unpickled_df.iloc[:, 0], list(X_pickle.index))

df3 = X_pickle.loc[X_pickle.index[list(bact_prot.index)]]
X = pd.concat([df1, df2, df3.iloc[:100, [12, 14, 15]]], axis=1)
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
nx.number_of_nodes(giantC)
###############################################################################
# Fit LASSO and perform cv to find alpha
# Author: Angyalka
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
alp = 10**np.linspace(1,-3,100)*0.5
np.min(alp)
np.max(alp)
ridge = Ridge(max_iter = 10000)
coefs = []

for a in alp:
    ridge.set_params(alpha=a)
    ridge.fit(scale(X_train), y_train)
    coefs.append(ridge.coef_)
    
ax = pyplot.gca()
ax.plot(alp*2, coefs)
ax.set_xscale('log')
pyplot.axis('tight')
pyplot.xlabel('alpha')
pyplot.ylabel('weights')
ax.set_title('Ridge Coefficients as a Function of Regularization', fontsize=16)

ridgecv = RidgeCV(alphas = alp, cv = 3)
ridgecv.fit(X_train, y_train)
#lassocv.alphas_
ridgecv.alpha_  # 5
ridge.set_params(alpha = ridgecv.alpha_)

ridge.fit(X_train, y_train)
train_pred_y = ridge.predict(X_train) 
test_pred_y = ridge.predict(X_test) 
    
print(f"train_MSE = {mean_squared_error(y_train, train_pred_y)}") # 0.05
print(f"test_MSE = {mean_squared_error(y_test, test_pred_y)}") # 0.126

c = pd.DataFrame(ridge.coef_, index = X.columns, columns= ['coefs'])
ridge_feature = c.index[np.nonzero(np.array(c))[0]]
ridge_coefs = c.iloc[np.nonzero(np.array(c))[0]]

len(c[c.coefs > 0.005])
ridge_res = c[c.coefs > 0.005]
'''
                                coefs
Average Centrality           1.912797
Modularity                   0.051938
Density                      1.912797
LCSG Denisty                 0.169768
LCSG Algebraic Connectivity  0.110522
num_68stars                  0.015519
num_70stars                  0.007922
num_71stars                  0.006164
num_86stars                  0.005738
num_98stars                  0.012881
num_105stars                 0.012881
'''
# Ridge feature importance plot
fig, ax = pyplot.subplots(figsize=(12, 7.5))
pyplot.rcParams['font.size'] = '12'
ax.barh(list(ridge_res.index), list(ridge_res.iloc[:,0]))
pyplot.xlabel('Coefficient')
pyplot.ylabel('Feature')
ax.set_title('Ridge Feature Importance', fontsize=16)
pyplot.show()

'''
LASSO did not like the multicollinearity of our data. 
All coefficients were zero, I suspected that no linear 
combination of any subset of the regressors would be useful 
for predicting the outcome.
'''
###############################################################################
# Fit and tune a random forest regressor tree
# Author: Angyalka
bag_forest_reg = ensemble.RandomForestRegressor(criterion="mse",
                                       n_jobs=-1,
                                       random_state=1)

bag_forest_reg.fit(X_train, y_train)

train_pred_y = bag_forest_reg.predict(X_train)
test_pred_y = bag_forest_reg.predict(X_test)

print(f"train_MSE = {mean_squared_error(y_train, train_pred_y)}") # 0.0057
print(f"test_MSE = {mean_squared_error(y_test, test_pred_y)}") # 0.029

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

print(f"train_MSE = {mean_squared_error(y_train, train_pred_y)}") # 0.005
print(f"test_MSE = {mean_squared_error(y_test, test_pred_y)}") # 0.028

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
    Modularity: 0.039
	Clique Count: 0.035
	LCSG Modularity: 0.032
	Average Centrality: 0.032
	Clique-Size Mean: 0.030
	Density: 0.029
	num_6stars: 0.028
	num_3stars: 0.028
	num_7stars: 0.026
	processing_time: 0.025
	num_1stars: 0.024
	critical_threshold: 0.022
	num_21stars: 0.022
	lcsg_nodes: 0.022
	Number of Triangles: 0.021
	GiantProportion: 0.021
	LCSG Denisty: 0.021
	num_2stars: 0.021
	LCSG Clique Count: 0.020
	LCSG Algebraic Connectivity: 0.020
	LCSG Degree Mean: 0.020
	num_15stars: 0.020
	LCSG Degree Max: 0.019
	num_16stars: 0.018
	num_5stars: 0.017
	LCSG Clique-Size Mean: 0.017
	LCSG Node Count: 0.016
	num_9stars: 0.016
	Number Edges Main Core: 0.016
	Clique-Size Max: 0.015
	num_14stars: 0.015
	num_4stars: 0.014
	num_12stars: 0.014
	num_27stars: 0.014
	num_56stars: 0.014
	num_19stars: 0.013
	num_20stars: 0.011
	LCSG Clique-Size Max: 0.011
	num_17stars: 0.010
'''
# Feature Importance plot
fig, ax = pyplot.subplots(figsize=(15, 12))
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

vis = pd.concat([df1, df3.iloc[:100, [12, 14, 15]]], axis = 1)
df1_vis = np.round(vis.T, 3)
df1_vis = df1_vis.iloc[:, :5]
print(df1_vis.to_latex(index=True))  

df1_vis = np.round(df1.head().T, 3)
print(df1_vis.to_latex(index=True)) 

df2_vis = np.round(df2.head().T.iloc[:6,:], 3)
print(df2_vis.to_latex(index=True)) 
