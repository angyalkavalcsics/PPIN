# -*- coding: utf-8 -*-
"""
@author: jhautala
"""

import sys
import pandas as pd
import pickle
from tqdm import tqdm
from time import sleep
import multiprocessing as mp
import queue
from memory_profiler import profile


# NOTE: change paths as needed
project_path = 'D:/classes/STA-596/project/'


# import local code (to satisfy requirements of parallelism)
sys.path.append(project_path)
import defs


# ---- constants
tmp_path = f'{project_path}tmp/'
data_path = f'{project_path}data/'
src_path = f'{data_path}/species_data.output.pickle'


# ---- functions
def calculate(
        in_df,
        sub_key,
        sub_value,
        num_items=-1,
        prior_item_count=0, # used to resume prior execution
    ):
    file_prefix = f'{tmp_path}/{sub_key}^{sub_value}'
    
    if prior_item_count == 0:
        prior_X = pd.DataFrame()
    else:
        prior_X = pd.read_pickle(f'{file_prefix}_{prior_item_count}.pickle')
    prior_item_count = prior_X.shape[0]
    
    # select subset
    sub_df = in_df.loc[in_df[sub_key] == sub_value]
    if num_items > 0:
        sub_df = sub_df.iloc[prior_item_count:num_items,:]
    else:
        sub_df = sub_df.iloc[prior_item_count:,:]

    
    num_items = sub_df.shape[0]
    out_path = f'{file_prefix}_{num_items}.pickle'
    
    cores = mp.cpu_count()
    num_workers = max(1, cores//2)
    
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
                pickle.dump(pd.concat([prior_X,X]), open(out_path, 'wb'))
                if not progress.update():
                    progress.refresh()
        except queue.Empty:
            sleep(.5)
    
    producer.join()
    for worker in workers:
        worker.join()
    consumer.join()
    progress.close()


@profile
def main():
    src_df = pd.read_pickle(f'{data_path}species_data.output.pickle')
    
    
    # ---- spot-calculate single subset
    # calculate(src_df,'Taxonomy Level 2','Bacteria_Proteobacteria')
    
    
    # ---- iterate over subsets
    sub_key = 'Taxonomy Level 2'
    for sub_value in src_df[sub_key].unique():
        calculate(src_df,sub_key,sub_value)
    
    
    # ---- copy from tmp to data
    # tmp_file_pattern = re.compile('(.+)\^(.+)_\d+\.pickle$')
    # for tmp_file in os.listdir(tmp_path):
    #     match = tmp_file_pattern.match(tmp_file)
    #     if match:
    #         out_file = f'{data_path}{match.group(1)}^{match.group(2)}.pickle'
    #         shutil.copy(f'{tmp_path}{tmp_file}', out_file)
            
    

if __name__ == "__main__":
    mp.freeze_support()
    main()
