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
import matplotlib.pyplot as plt
import networkx.algorithms.community as nx_comm
from networkx.algorithms.flow import shortest_augmenting_path
import statsmodels.api as sm
from sklearn import linear_model
from sklearn import ensemble
###############################################################################
# TODO: Definitely could reorganize a bit (my b).
# Starting with running through the adj matrices (once we settle on one) and 
# returning the adj for the giant comp rather than running this over and over 
# under each method. I'll work on this shortly (tmrw?) but if it bothers you then feel free!
###############################################################################
# Read in file
unpickled_df = pd.read_pickle("C:/Users/angya/OneDrive/Documents/species_data.output.pickle")
###############################################################################
# Trying to narrow things down
bact_prot = unpickled_df.loc[unpickled_df["Taxonomy Level 2"] == "Bacteria_Proteobacteria"]
# idk? randomly sample some of these
reduced_df = bact_prot.sample(n = 45)
adj = reduced_df.iloc[:, 8]

# backing up: 
# i think we do need bigger networks or 
# maybe just a variety of species

reduced_df2 = unpickled_df.sample(n=50)
adj2 = reduced_df2.iloc[:, 8]
###############################################################################
# Playing around with different drawings

first = adj2.iloc[31].todense()
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
# really unfortunate that its in the form of a dict
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
nx.is_connected(G)
[len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]

# Run this for subset of species and store values

res = pd.DataFrame({'Average Centrality' : [],
                    'Average Closed Triangles' : [],
                    'Modularity' : [],
                    'Node Connectivity' : []
                    })

# adj defined above
for i in range(len(adj)):
    temp = adj.iloc[i].todense()
    G = nx.convert_matrix.from_numpy_matrix(temp) 
    Gcc = max(nx.connected_components(G), key=len)
    giantC = G.subgraph(Gcc) 
    
    centrality = nx.degree_centrality(giantC)
    centrality_df = pd.DataFrame.from_dict(centrality, orient='index')
    avg_centrality = np.mean(centrality_df.iloc[:, 0])
    res.loc[i, 'Average Centrality'] = avg_centrality
    
    # really unfortunate that its in the form of a dict
    # eig_centrality = nx.eigenvector_centrality(G, max_iter=600)
    # eig_centrality_df = pd.DataFrame.from_dict(eig_centrality, orient='index')
    # avg_eig_centrality = np.mean(eig_centrality_df.iloc[:, 0])
    # res.loc[i, 'Average Eigencentrality'] = avg_eig_centrality
    
    triangles = nx.triangles(giantC)
    avg_triangles = np.mean(list(triangles.values()))
    res.loc[i, 'Average Closed Triangles'] = avg_triangles
    
    modularity = nx_comm.modularity(giantC, nx_comm.label_propagation_communities(giantC))
    res.loc[i, 'Modularity'] = modularity
    
    conn = nx.node_connectivity(giantC)
    res.loc[i, 'Node Connectivity'] = conn
    
######################
# 2

res2 = pd.DataFrame({'Average Centrality' : [],
                    'Average Closed Triangles' : [],
                    'Modularity' : [],
                    'Node Connectivity' : []
                    })

# adj defined above
for i in range(len(adj2)):
    temp = adj2.iloc[i].todense()
    G = nx.convert_matrix.from_numpy_matrix(temp) 
    Gcc = max(nx.connected_components(G), key=len)
    giantC = G.subgraph(Gcc) 
    
    centrality = nx.degree_centrality(giantC)
    centrality_df = pd.DataFrame.from_dict(centrality, orient='index')
    avg_centrality = np.mean(centrality_df.iloc[:, 0])
    res2.loc[i, 'Average Centrality'] = avg_centrality
    
    # really unfortunate that its in the form of a dict
    # eig_centrality = nx.eigenvector_centrality(G, max_iter=600)
    # eig_centrality_df = pd.DataFrame.from_dict(eig_centrality, orient='index')
    # avg_eig_centrality = np.mean(eig_centrality_df.iloc[:, 0])
    # res.loc[i, 'Average Eigencentrality'] = avg_eig_centrality
    
    triangles = nx.triangles(giantC)
    avg_triangles = np.mean(list(triangles.values()))
    res2.loc[i, 'Average Closed Triangles'] = avg_triangles
    
    modularity = nx_comm.modularity(giantC, nx_comm.label_propagation_communities(giantC))
    res2.loc[i, 'Modularity'] = modularity
    
    conn = nx.node_connectivity(giantC)
    res2.loc[i, 'Node Connectivity'] = conn

#######################
    
# Fit a linear model

# SLR with modularity as predictor R-sqrd = 0.954
X = sm.add_constant(res)
y = list(reduced_df['Evolution'])
model = sm.OLS(y, X.iloc[:, 3])
est = model.fit()
print(est.summary())

# SLR with modularity as predictor R-sqrd = 0.954
X2 = sm.add_constant(res2)
y2 = list(reduced_df2['Evolution'])
model2 = sm.OLS(y2, X2.iloc[:, 3])
est2 = model2.fit()
print(est.summary())

lassocv = linear_model.LassoCV(eps=.001,n_alphas=250,cv=10)
lasso_est = lassocv.fit(X,y)

# Fit a tree?

m = ensemble.BaggingRegressor()
m.fit(res,y)

yhat = m.predict(res)
np.mean((y - yhat)**2) # 0.007

feature_importances = np.mean([
    tree.feature_importances_ for tree in m.estimators_
], axis=0)

#############

m = ensemble.BaggingRegressor()
m.fit(res2,y2)

yhat2 = m.predict(res2)
np.mean((y2 - yhat2)**2) # 0.046

feature_importances = np.mean([
    tree.feature_importances_ for tree in m.estimators_
], axis=0)


#############

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
for a in range(len(adj2)):
    t = getstars(adj2.iloc[a])
    df_list.append(t)
    
stars = pd.concat(df_list, axis = 1)
stars.columns = reduced_df2['Species_ID']
stars = stars.fillna(0)

# Next: fit a LASSO separately and/or combine with clique df and fit 








