# -*- coding: utf-8 -*-
"""
@author: jhautala
"""

import os
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


# Cliques

@profile
def get_clique_count_df(G,as_df=False,maximal=False):
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

@profile
def get_degree_hist(G,as_df=False):
    degree = nx.degree_histogram(G)
    if as_df:
        degree_df = pd.DataFrame(degree).reset_index()
        degree_df.columns = ['degree', 'count']
        return degree_df
    else:
        return degree

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
    
    @profile
    def run(self):
        try:
            for i in range(self.df.shape[0]):
                while True:
                    try:
                        self.df_queue.put_nowait(self.df.iloc[[i]])
                        break
                    except queue.Full:
                        sleep(.5)
            for i in range(self.num_workers):
                self.df_queue.put(None)
        except KeyboardInterrupt:
            pass
    
class Worker(mp.Process):
    def __init__(self, df_queue, row_queue):
        super().__init__()
        self.daemon = True
        self.df_queue = df_queue
        self.row_queue = row_queue

    @profile
    def run(self):
        try:
            while True:
                try:
                    df = self.df_queue.get_nowait()
                    if df is None:
                        self.row_queue.put(None)
                        break
                    
                    # build row
                    row = {}
                    row['Species_ID'] = df['Species_ID'].iloc[0]
                    
                    # derive largest connected subgraph
                    ppin = df['Matrix'].iloc[0]
                    G = nx.convert_matrix.from_numpy_matrix(ppin.todense())
                    lcsg = G.subgraph(max(nx.connected_components(G), key=len))
                    
                    # centrality
                    centrality = nx.degree_centrality(lcsg)
                    centrality_df = pd.DataFrame.from_dict(centrality, orient='index')
                    row['avg_centrality'] = np.mean(centrality_df.iloc[:, 0])
                    
                    # triangles
                    triangles = nx.triangles(lcsg)
                    row['avg_triangles'] = np.mean(list(triangles.values()))
                    
                    # modularity and connectivity
                    row['modularity'] = nx_comm.modularity(
                        lcsg,
                        nx_comm.label_propagation_communities(lcsg))
                    # row['connectivity'] = nx.node_connectivity(lcsg, flow_func=shortest_augmenting_path)
                
                
                    # ---- clique stats
                    maximal_clique_counts = Counter()
                    for clique in nx.find_cliques(lcsg):
                        maximal_clique_counts[len(clique)] += 1
                    
                    maximal_df = pd.DataFrame.from_dict(maximal_clique_counts,orient='index').reset_index()
                    maximal_df.columns = ['clique_size', 'count']
                    maximal_stats = HistStats(maximal_df,'clique_size')
                    row['mc_count'] = maximal_stats.count
                    row['mc_max'] = maximal_stats.max
                    row['mc_mode'] = maximal_stats.mode
                    row['mc_mean'] = maximal_stats.mean
                    row.update({f'mc_{k}': v for k, v in maximal_clique_counts.items()})
                    
                    
                    # ---- exhaustive clique counts
                    # clique_counts = Counter()
                    # for clique in nx.enumerate_all_cliques(lcsg):
                    #     clique_counts[len(clique)] += 1
                    # row.update({f'c_{k}': v for k, v in clique_counts.items()})
                    # row.update({f'c_{k}': v for k, v in get_clique_count_df(lcsg).items()})
                    
                    
                    # ---- degree stats
                    degree_hist = nx.degree_histogram(lcsg)
                    degree_df = pd.DataFrame(degree_hist).reset_index()
                    degree_df.columns = ['degree', 'count']
                    degree_stats = HistStats(degree_df,'degree')
                    row['d_max'] = degree_stats.max
                    row['d_mode'] = degree_stats.mode
                    row['d_mean'] = degree_stats.mean
                    
                    # exhaustive degree counts
                    row.update({f'd_{k}': v for k, v in enumerate(degree_hist)})
                    # row.update({f'Node Degree {k}': v for k, v in enumerate(get_degree_hist(lcsg))})
                        
                    
                    # pet stat
                    row['giant_prop'] = lcsg.number_of_nodes()/G.number_of_nodes()
                    
                    # push DataFrame to queue
                    while True:
                        try:
                            self.row_queue.put_nowait(row)
                            break
                        except queue.Full:
                            sleep(.5)
                except queue.Empty:
                    sleep(.5)
        except KeyboardInterrupt:
            pass

class Consumer(mp.Process):
    def __init__(self, num_workers, row_queue, result_queue):
        super().__init__()
        self.daemon = True
        self.num_workers = num_workers
        self.row_queue = row_queue
        self.result_queue = result_queue
        self.df = pd.DataFrame()

    @profile
    def run(self):
        try:
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
        except KeyboardInterrupt:
            pass
