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
from sklearn import linear_model
from sklearn import ensemble
###############################################################################
# Read in file
unpickled_df = pd.read_pickle("yourpath/species_data.output.pickle")
###############################################################################
# Take a subset of the data 
bact_prot = unpickled_df.loc[unpickled_df["Taxonomy Level 2"] == "Bacteria_Proteobacteria"]
reduced_df = bact_prot.iloc[:75, :]
adj_train = reduced_df.iloc[:, 8]
###############################################################################
# Author: Angyalka
# Plot PPIN
n = 31 # controls which species we plot
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
# Create test data
# Author: Angyalka
'''
# We can't find test error because the number of k-stars would be different

test_df = bact_prot.iloc[76:100, :]
adj_test = test_df.iloc[:, 8]

predictors_test = get_df1(adj_test)
df1_test = pd.DataFrame(predictors_test, index = test_df['Species_ID'], columns=[
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

df2_test = get_df2(adj_test)
df2_test.index = test_df['Species_ID']

X_test = pd.concat([df1_test, df2_test], axis=1)
X_test['GiantProportion'] = X_test['LCSG Node Count']/X_test['Node Count']
y_test = test_df['Evolution']
'''
###############################################################################
# Fit LASSO
# Lasso fits an intercept by default
# Author: Angyalka
lasso = linear_model.Lasso(alpha = 0.6, max_iter = 10000)
mod = lasso.fit(X,y)
coefs = pd.DataFrame(mod.coef_, index = X.columns, columns= ['coefs'])
coefs.index[np.nonzero(np.array(coefs))[0]]

'''
Index(['Number of Triangles', 'Clique Count', 'LCSG Clique Count',
       'LCSG Node Count', 'LCSG Degree Max', 'num_1stars'],
      dtype='object')
'''

# Find training error

yhat = mod.predict(X)
np.mean((y - yhat)**2) # 0.05

###############################################################################
# Fit a tree
# Author: Angyalka
m = ensemble.RandomForestRegressor(max_depth=4)
m.fit(X,y)

# Find significant predictors
feature_importances = np.mean([
    tree.feature_importances_ for tree in m.estimators_
], axis=0)

for i in np.argsort(feature_importances)[::-1]:
    if feature_importances[i] < 0.01:
        break
    print(f'\t{X.columns[i]}: {feature_importances[i]:.3f}')
    
'''	
	num_1stars: 0.119
	Modularity: 0.106
	num_6stars: 0.091
	Clique Count: 0.063
	LCSG Clique-Size Mean: 0.044
	num_59stars: 0.044
	num_31stars: 0.034
	Average Centrality: 0.029
	num_15stars: 0.026
	num_14stars: 0.025
	GiantProportion: 0.024
	LCSG Clique-Size Max: 0.020
	num_2stars: 0.018
	num_18stars: 0.017
	num_3stars: 0.016
	num_7stars: 0.016
	Clique-Size Max: 0.015
	num_12stars: 0.015
	LCSG Clique Count: 0.015
	num_21stars: 0.014
	LCSG Degree Max: 0.014
	Number of Triangles: 0.011
	num_51stars: 0.011
	Clique-Size Mean: 0.011
	num_4stars: 0.011
	num_27stars: 0.011
	LCSG Node Count: 0.010
	num_13stars: 0.010
'''

# Find training error

yhat = m.predict(X)
np.mean((y - yhat)**2) # 0.009

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
