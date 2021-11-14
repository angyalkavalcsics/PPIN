# -*- coding: utf-8 -*-
"""
@author: jhautala
"""

import sys
import os
import re
import pandas as pd
import pickle
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import networkx.algorithms.community as nx_comm
from networkx.algorithms.flow import shortest_augmenting_path
import statsmodels.api as sm
import sklearn as sk
from sklearn import linear_model
from sklearn import ensemble
from sklearn import decomposition
import scipy as sp
from scipy.special import comb
from tqdm import tqdm
from time import sleep
from collections import Counter
import multiprocessing as mp
import queue
from memory_profiler import profile


path = 'D:/classes/STA-596/project/'
sys.path.append(path)
import defs

@profile
def main():
    in_df = pd.read_pickle(path + 'data/species_data.output.pickle')
    prior_X = pd.DataFrame()#pd.read_pickle(f'{path}/Bacteria_Proteobacteria_482.pickle')
    prior_item_count = prior_X.shape[0]
    
    # select subset
    sub_key = 'Taxonomy Level 2'
    sub_value = 'Bacteria_Proteobacteria'
    num_items = -1
    sub_df = in_df.loc[in_df[sub_key] == sub_value]
    if num_items > 0:
        sub_df = sub_df.iloc[prior_item_count:num_items,:]
    else:
        sub_df = sub_df.iloc[prior_item_count:,:]

    
    num_items = sub_df.shape[0]
    
    cores = mp.cpu_count()
    num_workers = 3#max(1, cores//2)
    
    progress = tqdm(total=num_items)
    df_queue = mp.Queue(num_workers*2)
    row_queue = mp.Queue(num_workers*2)
    result_queue = mp.Queue()

    # start producer
    producer = defs.Producer(sub_df, num_workers, df_queue)
    producer.start()
    
    # start workers
    workers = []
    for i in (range(num_workers)):
        worker = defs.Worker(df_queue, row_queue)
        worker.start()
        workers.append(worker)
    
    # start consumer
    consumer = defs.Consumer(
        num_workers,
        row_queue,
        result_queue,
        )
    consumer.start()
    
    # get progress updates
    while True:
        try:
            X = result_queue.get_nowait()
            if X is None:
                break
            else:
                pickle.dump(pd.concat([X,prior_X]), open(f'{path}{sub_value}_{num_items}.pickle', 'wb'))
                if not progress.update():
                    progress.refresh()
        except queue.Empty:
            sleep(.5)
    
    producer.join()
    for worker in workers:
        worker.join()
    consumer.join()
    progress.close()

if __name__ == "__main__":
    mp.freeze_support()
    main()
    
