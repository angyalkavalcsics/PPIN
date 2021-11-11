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
import sklearn as sk
from sklearn import linear_model
from sklearn import ensemble
from sklearn import decomposition
from tqdm import tqdm
###############################################################################
# TODO: 
# Find cliques for each network.
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
adj = reduced_df['Matrix']
###############################################################################
# Some plots
first = adj.iloc[31].todense()
G = nx.convert_matrix.from_numpy_matrix(first) 
# need to fix the networks so we only take the 
# giant connected component from each one 
Gcc = max(nx.connected_components(G), key=len)
giantC = G.subgraph(Gcc) 
nx.draw(giantC, node_size = 20)
'''
The fancy layouts do not really work well 
for disconnected graphs because the disconnected 
components tend to "drift away" from each other.
Which is why I didn't notice intially that the 
graphs are all disconnected. '
Try nx.draw(G, node_size = 25)
to see the difference.
'''
#ax.set_title('species name')
kawai = nx.kamada_kawai_layout(G)
nx.draw(G, kawai, node_size=25)

spring = nx.spring_layout(G, k=0.95, iterations=200)
nx.draw(G, spring, node_size=30) # dense!

###############################################################################
# Seeing how these internal functions work 

# Degree Centrality

'''
Degree centrality is the simplest centrality measure to compute. Recall that a 
node's degree is simply a count of how many connections (i.e., edges) 
it has. The degree centrality for a node is simply its degree.
'''

centrality = nx.degree_centrality(G)
#print(['%s %0.2f'%(node,centrality[node]) for node in centrality])
centrality_df = pd.DataFrame.from_dict(centrality, orient='index')
avg_centrality = np.mean(centrality_df.iloc[:, 0])

# Eigencentrality

'''
In graph theory, eigenvector centrality is a measure of the influence of a node 
in a network. Relative scores are assigned to all nodes in the network based on 
the concept that connections to high-scoring nodes contribute more to the score 
of the node in question than equal connections to low-scoring nodes. A high 
eigenvector score means that a node is connected to many nodes who themselves 
have high scores.
'''

eig_centrality = nx.eigenvector_centrality(G)
eig_centrality_df = pd.DataFrame.from_dict(eig_centrality, orient='index')
avg_eig_centrality = np.mean(eig_centrality_df.iloc[:, 0])

# Number of triangles

'''
When computing triangles for the entire graph each triangle is counted three 
times, once at each node. 
Returns: Number of triangles keyed by node label.
'''
triangles = nx.triangles(G)
avg_triangles = np.mean(list(triangles.values()))

# Modularity

'''
Networks with high modularity have dense connections between 
the nodes within modules but sparse connections between nodes 
in different modules. 
'''

modularity = nx_comm.modularity(G, nx_comm.label_propagation_communities(G))

# node connectivity
nx.node_connectivity(G, flow_func=shortest_augmenting_path)

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

'''
If you are wondering why I didn't include eigencentrality as a predictor it
is because the function would not converge occassionally. I tried setting
max iterations to like 1000 then just decided to leave it alone.

Also I took out node connectivity since it is 1 for each network.
'''

predictors = []
for i in tqdm(range(len(adj))):
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
        degree_stats.max,
        degree_stats.mode,
        degree_stats.mean,
        lcsg_degree_stats.count,
        lcsg_degree_stats.max,
        lcsg_degree_stats.mode,
        lcsg_degree_stats.mean,
        ])

###############################################################################

X = sm.add_constant(pd.DataFrame(predictors, columns=[
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
    'Degree Max',
    'Degree Mode',
    'Degree Mean',
    'LCSG Node Count',
    'LCSG Degree Max',
    'LCSG Degree Mode',
    'LCSG Degree Mean',
    ]))
y = reduced_df['Evolution']

'''
Trying some meta-stats, it seems like 'GiantProportion' (the proportion of
total nodes that are included in the largest connected subgraph) is almost
as powerful as modularity (as meaured by SLR R^2). We might expect this is
another way to measure connectivity/resiliency of the proteome.
'''
X['GiantProportion'] = X['LCSG Node Count']/X['Node Count']

compare = {
    'Modularity',
    'GiantProportion',
    'ModPerNode',
    'Node Count',
    # 'Degree Max',
    # 'Degree Mode',
    # 'Degree Mean',
    'LCSG Node Count',
    # 'LCSG Degree Max',
    # 'LCSG Degree Mode',
    # 'LCSG Degree Mean',
    # 'Clique Count',
    # 'Clique-Size Max',
    # 'Clique-Size Mode',
    # 'Clique-Size Mean',
    # 'LCSG Clique Count',
    # 'LCSG Clique-Size Max',
    # 'LCSG Clique-Size Mode',
    # 'LCSG Clique-Size Mean',
    }
sns.pairplot(X[compare], kind="reg", diag_kind="kde")

# Fit a linear model
# SLR with modularity as predictor
for col in X.columns[1:]:
    # if col != 'Modularity': continue
    print('\n')
    model = sm.OLS(y, X[col])
    est = model.fit()
    print(est.summary())

###############################################################################
# Fit LASSO

lassocv = linear_model.LassoCV(eps=.001,n_alphas=250,cv=10)
lasso_est = lassocv.fit(X,y)

###############################################################################
# Fit a tree

m = ensemble.BaggingRegressor()
m.fit(X,y)

yhat = m.predict(X)
np.mean((y - yhat)**2) # 0.007

feature_importances = np.mean([
    tree.feature_importances_ for tree in m.estimators_
], axis=0)

for i in np.argsort(feature_importances)[::-1]:
    if feature_importances[i] == 0:
        break
    print(f'\t{X.columns[i]}: {feature_importances[i]:.3f}')

###############################################################################

# s_k as predictor

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

df_list = []
for a in tqdm(range(len(adj))):
    t = getstars(adj.iloc[a])
    df_list.append(t)
    
stars = pd.concat(df_list, axis = 1)
stars.columns = reduced_df['Species_ID']
stars = stars.fillna(0)

# Next: fit a LASSO separately and/or combine with clique df and fit 

