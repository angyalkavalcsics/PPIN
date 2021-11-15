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
# TODO: 
# Run LASSO on some combination of the statistics we found.
# Can we find any combination that beats modularity as a predictor?
# Plots?
###############################################################################
# Read in file
unpickled_df = pd.read_pickle("C:/Users/angya/OneDrive/Documents/species_data.output.pickle")
###############################################################################
# Take a subset of the data

bact_prot = unpickled_df.loc[unpickled_df["Taxonomy Level 2"] == "Bacteria_Proteobacteria"]
reduced_df = bact_prot.iloc[:75, :]
adj_train = reduced_df.iloc[:, 8]
###############################################################################
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

# Run this for subset of species and store values
def get_df1(adj):
    predictors = []
    for i in range(len(adj)):
        temp = adj.iloc[i].todense()
        G = nx.convert_matrix.from_numpy_matrix(temp) 
        Gcc = max(nx.connected_components(G), key=len)
        giantC = G.subgraph(Gcc) 
        
        centrality = nx.degree_centrality(giantC)
        centrality_df = pd.DataFrame.from_dict(centrality, orient='index')
        avg_centrality = np.mean(centrality_df.iloc[:, 0])
        
        triangles = nx.triangles(giantC)
        avg_triangles = np.mean(list(triangles.values()))
        
        modularity = nx_comm.modularity(giantC, 
                                        nx_comm.label_propagation_communities(giantC))
        
        clique_size = HistStats(get_clique_count_df(G), 'clique_size')
        lcsg_clique_size = HistStats(get_clique_count_df(giantC), 'clique_size')
        
        degree_stats = HistStats(get_degree_hist(G), 'degree')
        lcsg_degree_stats = HistStats(get_degree_hist(giantC), 'degree')
        
        predictors.append([
            avg_centrality,
            avg_triangles,
            modularity,
            clique_size.count,
            clique_size.max,
            clique_size.mode,
            clique_size.mean,
            lcsg_clique_size.count,
            lcsg_clique_size.max,
            lcsg_clique_size.mode,
            lcsg_clique_size.mean,
            degree_stats.count,
            lcsg_degree_stats.count,
            lcsg_degree_stats.max,
            lcsg_degree_stats.mode,
            lcsg_degree_stats.mean,
            ])
    return(predictors)

###############################################################################
# Stars

'''
The S_k is a node with k edges
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

predictors = get_df1(adj_train)
df1 = pd.DataFrame(predictors, index = reduced_df['Species_ID'], columns=[
    'Average Centrality',
    'Average Closed Triangles',
    'Modularity',
    'Clique Count',
    'Clique-Size Max',
    'Clique-Size Mode',
    'Clique-Size Mean',
    'LCSG Clique Count',
    'LCSG Clique-Size Max',
    'LCSG Clique-Size Mode',
    'LCSG Clique-Size Mean',
    'Node Count',
    'LCSG Node Count',
    'LCSG Degree Max',
    'LCSG Degree Mode',
    'LCSG Degree Mean',
    ])

###############################################################################
# Combine data frames of the various stats

df2 = get_df2(adj_train)
df2.index = reduced_df['Species_ID']

X = pd.concat([df1, df2], axis=1)

y = reduced_df['Evolution']

'''
Trying some meta-stats, it seems like 'GiantProportion' (the proportion of
total nodes that are included in the largest connected subgraph) is almost
as powerful as modularity (as meaured by SLR R^2). We might expect this is
another way to measure connectivity/resiliency of the proteome.
'''

X['GiantProportion'] = X['LCSG Node Count']/X['Node Count']

###############################################################################
'''
On the prelim he says:
    
After comparing the two methods, you should perform a full analysis of the 
data including error estimates and other relevant quantities. You should 
include relevant tables and figures.

Notes:
* We have one figure above (the graph) and the pairplot below. 
* I could not get cross validation to work for LASSO -- convergence issue.
* I found the training error 
'''
###############################################################################
# Create test data
'''
# We can't find test error because the number of k-stars would be different

test_df = bact_prot.iloc[76:100, :]
adj_test = test_df.iloc[:, 8]

predictors_test = get_df1(adj_test)
df1_test = pd.DataFrame(predictors_test, index = test_df['Species_ID'], columns=[
    'Average Centrality',
    'Average Closed Triangles',
    'Modularity',
    'Clique Count',
    'Clique-Size Max',
    'Clique-Size Mode',
    'Clique-Size Mean',
    'LCSG Clique Count',
    'LCSG Clique-Size Max',
    'LCSG Clique-Size Mode',
    'LCSG Clique-Size Mean',
    'Node Count',
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
lasso = linear_model.Lasso(alpha = 0.6, max_iter = 10000)
mod = lasso.fit(X,y)
coefs = pd.DataFrame(mod.coef_, index = X.columns, columns= ['coefs'])
coefs.index[np.nonzero(np.array(coefs))[0]]

'''
Index(['Average Closed Triangles', 'Clique Count', 'LCSG Clique Count',
       'Node Count', 'LCSG Node Count', 'num_1stars'],
      dtype='object')
'''

# Find training error

yhat = mod.predict(X)
np.mean((y - yhat)**2) # 0.046

###############################################################################
# Fit a tree
m = ensemble.BaggingRegressor()
m.fit(X,y)

# Find significant predictors

feature_importances = np.mean([
    tree.feature_importances_ for tree in m.estimators_
], axis=0)

for i in np.argsort(feature_importances)[::-1]:
    if feature_importances[i] == 0:
        break
    print(f'\t{X.columns[i]}: {feature_importances[i]:.3f}')
    
'''
    num_6stars: 0.165
	Clique Count: 0.161
	Node Count: 0.077
	Modularity: 0.066
	num_1stars: 0.065
	LCSG Clique-Size Mean: 0.061
	num_59stars: 0.045
	num_15stars: 0.039
	num_4stars: 0.030
	num_12stars: 0.020
	num_2stars: 0.019
	LCSG Degree Mean: 0.018
	LCSG Degree Max: 0.018
	Clique-Size Max: 0.017
	GiantProportion: 0.016
	LCSG Clique-Size Max: 0.016
	num_8stars: 0.015
	num_7stars: 0.014
	num_11stars: 0.011
	num_28stars: 0.009
	num_9stars: 0.009
	num_22stars: 0.008
	num_25stars: 0.008
	num_27stars: 0.008
	num_10stars: 0.007
	num_23stars: 0.007
	num_16stars: 0.007
	num_58stars: 0.005
	num_38stars: 0.005
	Clique-Size Mean: 0.005
	num_63stars: 0.004
	num_37stars: 0.003
	num_18stars: 0.003
	LCSG Node Count: 0.003
	LCSG Clique-Size Mode: 0.003
	Average Closed Triangles: 0.002
	num_41stars: 0.002
	num_53stars: 0.002
	Average Centrality: 0.002
	num_36stars: 0.002
	num_21stars: 0.002
	num_76stars: 0.002
	LCSG Clique Count: 0.002
	num_30stars: 0.001
	num_39stars: 0.001
	num_31stars: 0.001
	num_19stars: 0.001
	num_17stars: 0.001
	LCSG Degree Mode: 0.001
	num_3stars: 0.001
	num_5stars: 0.001
	num_33stars: 0.001
	num_55stars: 0.001
	num_20stars: 0.001
	num_14stars: 0.001
	num_40stars: 0.000
	num_29stars: 0.000
	num_48stars: 0.000
	num_35stars: 0.000
	num_44stars: 0.000
	num_13stars: 0.000
	num_45stars: 0.000
	num_62stars: 0.000
	num_34stars: 0.000
	num_50stars: 0.000
	num_60stars: 0.000
	num_26stars: 0.000
	num_56stars: 0.000
	num_24stars: 0.000
	num_54stars: 0.000
	num_47stars: 0.000
	Clique-Size Mode: 0.000
	num_43stars: 0.000
	num_51stars: 0.000
	num_61stars: 0.000
	num_32stars: 0.000
	num_46stars: 0.000
	num_77stars: 0.000
	num_52stars: 0.000
	num_66stars: 0.000
'''

# Find training error

yhat = m.predict(X)
np.mean((y - yhat)**2) # 0.0078

###############################################################################
# Pairs plot to show relationship between the common significant 
# predictors from both models

compare = {
    'Modularity',
    'num_1stars',
    'Node Count',
    'LCSG Node Count',
    }

pp = sns.pairplot(X[compare], kind="reg", diag_kind="kde")
pp.fig.suptitle("Pairwise Relationships", y=1.00)

###############################################################################

















