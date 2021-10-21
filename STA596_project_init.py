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

# Playing around with different drawings
# my vote is for kawai
kawai = nx.kamada_kawai_layout(G)
nx.draw(G, kawai, node_size=40)

spring = nx.spring_layout(G, k=0.9, iterations=100)
nx.draw(G, spring, node_size=50) # dense!

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
print(['%s %0.2f'%(node,centrality[node]) for node in centrality])
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
print(['%s %0.2f'%(node,eig_centrality[node]) for node in eig_centrality])
eig_centrality_df = pd.DataFrame.from_dict(eig_centrality, orient='index')
avg_eig_centrality = np.mean(eig_centrality_df.iloc[:, 0])

# Number of triangles

'''
When computing triangles for the entire graph each triangle is counted three 
times, once at each node. 

Returns: Number of triangles keyed by node label.
'''
triangles = nx.triangles(G)
avg_triangles = np.mean(np.asarray(triangles.values()))

























