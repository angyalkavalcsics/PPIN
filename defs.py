# -*- coding: utf-8 -*-
"""
@author: jhautala
"""

import os
import sys
import copy
import random
import pandas as pd
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
from networkx.algorithms.flow import shortest_augmenting_path
from time import sleep
from collections import Counter
import multiprocessing as mp
import queue
import tqdm
from memory_profiler import profile
import time
from datetime import datetime



# Cliques

# @profile
def get_clique_count_df(G,as_df=True,maximal=True):
    clique_counts = Counter()
    for clique in nx.find_cliques(G) if maximal else nx.enumerate_all_cliques(G):
        clique_counts[len(clique)] += 1
    
    if as_df:
        # convert to pandas series
        clique_count_df = pd.DataFrame.from_dict(clique_counts,orient='index').reset_index()
        clique_count_df.columns = ['clique_size', 'count']
        return clique_count_df
    else:
        return clique_counts

# Degree

# @profile
def get_degree_hist(G,as_df=False):
    degree = nx.degree_histogram(G)
    if as_df:
        degree_df = pd.DataFrame(degree).reset_index()
        degree_df.columns = ['degree', 'count']
        return degree_df
    else:
        return degree

def getstars(G):
    A = nx.adjacency_matrix(G).todense()
    deg = np.asarray(np.sum(A,axis=0)).flatten()
    values, counts = np.unique(deg, return_counts=True)
    stars_sm = pd.DataFrame(counts)
    stars_sm = stars_sm.set_index(values)
    return(stars_sm)

# Critical Threshold

def modifiable_lcsg(G):
    return nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))

def remove_edges(G,n=1):
    if G.number_of_edges() <= n:
        return False
    G.remove_edges_from(random.choices(list(G.edges()),k=n))
    return True

def remove_nodes(G,n=1):
    if G.number_of_nodes() <= n:
        return False
    G.remove_nodes_from(random.choices(list(G),k=n))
    return True

def simulate_failure(
        init_G,
        max_rm=1,
        min_lcsg=.1,
        num_iter=10,
        use_edges=False,
        ):
    min_rm = sys.maxsize
    max_rm = -sys.maxsize
    total_rm = 0
    remove = remove_edges if use_edges else remove_nodes
    for curr_iter in range(num_iter):
        G = nx.Graph(init_G)
        rm_count = 0
        while True:
            curr_rm = random.randint(1,max_rm) if max_rm > 1 else 1
            rm_count += curr_rm
            
            # destroy!
            if not remove(G,curr_rm):
                break # can't remove any more!
            
            # remove chaff
            chaff = [node for node,degree in dict(G.degree()).items() if degree == 0]
            rm_count += len(chaff)
            G.remove_nodes_from(chaff)
            
            max_conn = max(nx.connected_components(G), key=len, default=set())
            if len(max_conn)/G.number_of_nodes() < min_lcsg:
                break # network failure
        total_rm += rm_count
        min_rm = min(min_rm,rm_count)
        max_rm = max(max_rm,rm_count)
    return min_rm, max_rm, total_rm/num_iter


# Overall calculation

def calculate(G,graph_cases,row={}):
    for (prefix, graph) in graph_cases:
        # ---- centrality
        centrality = nx.degree_centrality(graph)
        centrality_df = pd.DataFrame.from_dict(centrality, orient='index')
        row[f'{prefix}avg_centrality'] = np.mean(centrality_df.iloc[:, 0])
        
        
        # ---- triangles
        triangles = nx.triangles(graph)
        row[f'{prefix}avg_triangles'] = np.mean(list(triangles.values()))
        
        
        # ---- modularity and connectivity
        row[f'{prefix}modularity'] = nx_comm.modularity(graph, nx_comm.label_propagation_communities(graph))
        row[f'{prefix}connectivity'] = nx.node_connectivity(graph, flow_func=shortest_augmenting_path)
        
        
        # ---- density
        n = graph.number_of_nodes()
        e = graph.number_of_edges()
        density = 2*e/(n*(n-1))
        row[f'{prefix}nodes'] = n
        row[f'{prefix}edges'] = e
        row[f'{prefix}density'] = density
        
        
        # ---- clique stats
        maximal_clique_counts = Counter()
        for clique in nx.find_cliques(graph):
            maximal_clique_counts[len(clique)] += 1
        maximal_clique_df = pd.DataFrame.from_dict(maximal_clique_counts,orient='index').reset_index()
        maximal_clique_df.columns = ['clique_size', 'count']
        maximal_clique_stats = HistStats(maximal_clique_df,'clique_size')
        row[f'{prefix}clique_count'] = maximal_clique_stats.count
        row[f'{prefix}largest_clique'] = maximal_clique_stats.max
        row[f'{prefix}clique_mode'] = maximal_clique_stats.mode
        row[f'{prefix}avg_clique'] = maximal_clique_stats.mean
        
        # #      exhaustive clique counts
        # if graph is G:
        #     row.update({f'mc_{k}': v for k, v in maximal_clique_counts.items()})
    
    
        # ---- degree stats (equivalent to getstars)
        degree_hist = nx.degree_histogram(graph)
        degree_df = pd.DataFrame(degree_hist).reset_index()
        degree_df.columns = ['degree', 'count']
        degree_stats = HistStats(degree_df,'degree')
        row[f'{prefix}max_degree'] = degree_stats.max
        row[f'{prefix}degree_mode'] = degree_stats.mode
        row[f'{prefix}avg_degree'] = degree_stats.mean
        
        # #      exhaustive degree counts
        # if graph is G:
        #     row.update({f'd_{k}': v for k, v in enumerate(degree_hist)})
    
    
        # ---- failure via node removal
        min_rm, max_rm, mean_rm = simulate_failure(graph)
        row[f'{prefix}min_critical'] = min_rm
        row[f'{prefix}max_critical'] = max_rm
        row[f'{prefix}avg_critical'] = mean_rm
        
        # ---- failure via edge removal
        min_rm, max_rm, mean_rm = simulate_failure(graph,use_edges=True)
        row[f'{prefix}min_critical'] = min_rm
        row[f'{prefix}max_critical'] = max_rm
        row[f'{prefix}avg_critical'] = mean_rm

# classes

class HistStats:
    def __init__(self, df, val):
        self.count = df['count'].sum()
        self.max = df[val].max()
        self.mode = df.iloc[df['count'].idxmax()][val]
        self.mean = (df[val] * df['count']).sum() / self.count

class Producer(mp.Process):
    def __init__(
            self,
            df,
            num_workers,
            df_queue,
            ):
        super().__init__()
        self.daemon = True
        self.df = df
        self.num_workers = num_workers
        self.df_queue = df_queue
    
    # @profile
    def run(self):
        for i in range(self.df.shape[0]):
            while True:
                try:
                    self.df_queue.put_nowait(self.df.iloc[[i]])
                    break
                except queue.Full:
                    sleep(.5)
        for i in range(self.num_workers):
            self.df_queue.put(None)

class Worker(mp.Process):
    def __init__(self, df_queue, row_queue):
        super().__init__()
        self.daemon = True
        self.df_queue = df_queue
        self.row_queue = row_queue

    # @profile
    def run(self):
        pid = os.getpid()
        while True:
            try:
                df = self.df_queue.get_nowait()
                if df is None:
                    self.row_queue.put(None)
                    break
                                
                # build row
                row = {}
                row['Species_ID'] = df.index[0]
                
                start = time.time()
                
                # determine giant component
                ppin = df['Matrix'].iloc[0]
                G = nx.convert_matrix.from_numpy_matrix(ppin.todense())
                lcsg = G.subgraph(max(nx.connected_components(G), key=len))
                graph_cases = [('',G),('lcsg_',lcsg)]
                
                
                calculate(G,graph_cases,row)
                
                
                # pet stat
                row['proportion_in_giant'] = lcsg.number_of_nodes()/G.number_of_nodes()
                
                
                # measure performance
                row['processing_time'] = time.time() - start
                
                
                # end timestamp
                # row['end_timestamp'] = datetime.now().strftime("%H:%M:%S")
                
                
                # push DataFrame to queue
                while True:
                    try:
                        self.row_queue.put_nowait(row)
                        break
                    except queue.Full:
                        sleep(.5)
            except queue.Empty:
                sleep(.5)

class Consumer(mp.Process):
    def __init__(self, num_workers, row_queue, result_queue):
        super().__init__()
        self.daemon = True
        self.num_workers = num_workers
        self.row_queue = row_queue
        self.result_queue = result_queue
        self.df = pd.DataFrame()
    
    # @profile
    def run(self):
        remaining_workers = self.num_workers
        while True:
            try:
                row = self.row_queue.get_nowait()
                if row is None:
                    remaining_workers -= 1
                    if remaining_workers == 0:
                        self.result_queue.put(None)
                        break
                else:
                    self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
                    self.result_queue.put(self.df)
            except queue.Empty:
                sleep(.5)
