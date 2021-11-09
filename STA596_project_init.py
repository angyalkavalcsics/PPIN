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
import matplotlib.pyplot as pyplot
import networkx.algorithms.community as nx_comm
from networkx.algorithms.flow import shortest_augmenting_path
import statsmodels.api as sm
from sklearn import linear_model
from sklearn import ensemble
from tqdm import tqdm
from time import sleep
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
adj = reduced_df.iloc[:, 8]
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

# Run this for subset of species and store values

'''
If you are wondering why I didn't include eigencentrality as a predictor it
is because the function would not converge occassionally. I tried setting
max iterations to like 1000 then just decided to leave it alone.

Also I took out node connectivity since it is 1 for each network.
'''

res = pd.DataFrame({'Average Centrality' : [],
                    'Average Closed Triangles' : [],
                    'Modularity' : []
                    })

for i in tqdm(range(len(adj))):
    temp = adj.iloc[i].todense()
    G = nx.convert_matrix.from_numpy_matrix(temp) 
    Gcc = max(nx.connected_components(G), key=len)
    giantC = G.subgraph(Gcc) 
    
    centrality = nx.degree_centrality(giantC)
    centrality_df = pd.DataFrame.from_dict(centrality, orient='index')
    avg_centrality = np.mean(centrality_df.iloc[:, 0])
    res.loc[i, 'Average Centrality'] = avg_centrality
    
    triangles = nx.triangles(giantC)
    avg_triangles = np.mean(list(triangles.values()))
    res.loc[i, 'Average Closed Triangles'] = avg_triangles
    
    modularity = nx_comm.modularity(giantC, 
                                    nx_comm.label_propagation_communities(giantC))
    res.loc[i, 'Modularity'] = modularity
    sleep(3)
    
###############################################################################
    
# Fit a linear model

# SLR with modularity as predictor 
X = sm.add_constant(res)
y = list(reduced_df['Evolution'])
model = sm.OLS(y, X.iloc[:, 2])
est = model.fit()
print(est.summary())

###############################################################################
# Fit LASSO

lassocv = linear_model.LassoCV(eps=.001,n_alphas=250,cv=10)
lasso_est = lassocv.fit(X,y)

###############################################################################
# Fit a tree

m = ensemble.BaggingRegressor()
m.fit(res,y)

yhat = m.predict(res)
np.mean((y - yhat)**2) # 0.007

feature_importances = np.mean([
    tree.feature_importances_ for tree in m.estimators_
], axis=0)

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
    sleep(0.05)
    
stars = pd.concat(df_list, axis = 1)
stars.columns = reduced_df['Species_ID']
stars = stars.fillna(0)

# Next: fit a LASSO separately and/or combine with clique df and fit 

