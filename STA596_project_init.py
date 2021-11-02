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
import matplotlib.cbook
###############################################################################
# Read in file
unpickled_df = pd.read_pickle("C:/Users/angya/OneDrive/Documents/species_data.output.pickle")
# Get all adjacency matrices
adj = unpickled_df.iloc[:, 8]
# Test network species ID: 882 
# should have 7520 edges
first = adj[0].todense()
G = nx.convert_matrix.from_numpy_matrix(first) 
###############################################################################
# Trying to narrow things down
bact_prot = unpickled_df.loc[unpickled_df["Taxonomy Level 2"] == "Bacteria_Proteobacteria"]
# idk? randomly sample some of these
df_reduced = bact_prot.sample(n = 45)
adj = df_reduced.iloc[:, 8]

first = adj.iloc[44].todense()
G = nx.convert_matrix.from_numpy_matrix(first) 
###############################################################################
# Playing around with different drawings
# my vote is for kawai
#ax.set_title('species name')
kawai = nx.kamada_kawai_layout(G)
nx.draw(G, kawai, node_size=25)

spring = nx.spring_layout(G, k=0.95, iterations=200)
nx.draw(G, spring, node_size=30) # dense!

from networkx.drawing.nx_agraph import graphviz_layout
from pylab import rcParams
# if someone could make this work?
# pip install pygraphviz
viz = graphviz_layout(G, prog='dot')
nx.draw(G, viz)
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
import networkx.algorithms.community as nx_comm
modularity = nx_comm.modularity(G, nx_comm.label_propagation_communities(G))


# Run this for subset of species and store values

res = pd.DataFrame({'Average Centrality' : [],
                    'Average Eigencentrality' : [],
                    'Average Closed Triangles' : [],
                    'Modularity' : []
                    })
# adj defined on line 19
for i in range(len(adj)):
    temp = adj.iloc[i].todense()
    G = nx.convert_matrix.from_numpy_matrix(temp) 
    
    centrality = nx.degree_centrality(G)
    centrality_df = pd.DataFrame.from_dict(centrality, orient='index')
    avg_centrality = np.mean(centrality_df.iloc[:, 0])
    res.loc[i, 'Average Centrality'] = avg_centrality
    
    # really unfortunate that its in the form of a dict
    eig_centrality = nx.eigenvector_centrality(G, max_iter=600)
    eig_centrality_df = pd.DataFrame.from_dict(eig_centrality, orient='index')
    avg_eig_centrality = np.mean(eig_centrality_df.iloc[:, 0])
    res.loc[i, 'Average Eigencentrality'] = avg_eig_centrality
    
    triangles = nx.triangles(G)
    avg_triangles = np.mean(list(triangles.values()))
    res.loc[i, 'Average Closed Triangles'] = avg_triangles
    
    modularity = nx_comm.modularity(G, nx_comm.label_propagation_communities(G))
    res.loc[i, 'Modularity'] = modularity
    
    
# Fit a linear model

import statsmodels.api as sm
X = sm.add_constant(res)
y = list(df_reduced['Evolution'])
model = sm.OLS(y, X)
est = model.fit()
print(est.summary())

















