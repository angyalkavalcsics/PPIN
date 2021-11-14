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

path = 'D:/classes/STA-596/project/'
sys.path.append(path)
import defs


prior_X = pd.read_pickle(f'{path}/Bacteria_Proteobacteria_10.pickle')

def main():
    in_df = pd.read_pickle(path + 'data/species_data.output.pickle')
    
    # select subset
    sub_key = 'Taxonomy Level 2'
    sub_value = 'Bacteria_Proteobacteria'
    num_items = -1
    sub_df = in_df.loc[in_df[sub_key] == sub_value]
    if num_items > 0:
        sub_df = sub_df.iloc[:num_items, :]
    else:
        num_items = sub_df.shape[0]
    ppins = sub_df['Matrix']
    
    cores = mp.cpu_count()
    num_workers = max(1, cores//2)
    
    progress = tqdm(total=num_items)
    ppin_queue = mp.Queue(num_workers*2)
    row_queue = mp.Queue(num_workers*2)
    result_queue = mp.Queue()
    update_queue = mp.Queue()

    # start producer
    producer = defs.Producer(ppins,  num_workers, ppin_queue)
    producer.start()
    
    # start workers
    workers = []
    for i in (range(num_workers)):
        worker = defs.Worker(ppin_queue, row_queue)
        worker.start()
        workers.append(worker)
        
    # start consumer
    consumer = defs.Consumer(
        num_workers,
        row_queue,
        result_queue,
        update_queue,
        )
    consumer.start()
    
    # get progress updates
    while True:
        try:
            item = update_queue.get_nowait()
            if item is None:
                progress.close()
                break
            else:
                if not progress.update():
                    progress.refresh()
        except queue.Empty:
            sleep(.5)
    
    producer.join()
    for worker in workers:
        worker.join()
    consumer.join()
    
    X = result_queue.get()
    
    progress.close()
    
    X['Species_ID'] = sub_df.reset_index()['Species_ID']
    pickle.dump(X, open(f'{path}{sub_value}_{num_items}.pickle', 'wb'))

if __name__ == "__main__":
    mp.freeze_support()
    main()
    
