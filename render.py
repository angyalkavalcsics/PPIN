# -*- coding: utf-8 -*-
"""
@author: jhautala
"""

import os
import sys
import glob
import random
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import imageio
from PIL import Image


# ---- config
# NOTE: change to match project location
project_path = 'D:/classes/STA-596/project/'
data_path = f'{project_path}data/'
tmp_path = f'{project_path}tmp/'


# ---- functions
def load_orig():
    df = pd.read_pickle(f'{data_path}species_data.output.pickle')
    return df.set_index('Species_ID')

def load_data():
    ''' This function loads and merges all the predictors and response with
    'Species_ID' as index. '''
    X = pd.read_pickle(f'{data_path}predictors.pickle')
    y = pd.read_pickle(f'{data_path}response.pickle')
    return X, y, pd.merge(X, y, on='Species_ID')

def load_subset():
    merged = pd.read_pickle(f'{data_path}Taxonomy Level 2^Bacteria_Proteobacteria.pickle')
    X = merged.drop(columns='Evolution')
    y = merged['Evolution']
    return X, y, merged

def get_extreme(df,col):
    tmp_min = df.loc[df[col] == df[col].min()].index[0]
    tmp_max = df.loc[df[col] == df[col].max()].index[0]
    return tmp_min, tmp_max

# ---- execution

src_df = load_orig()
X, y, merged = load_subset()


# add graphs
graphs = src_df.loc[src_df['Taxonomy Level 2'] == 'Bacteria_Proteobacteria']
graphs = graphs['Matrix']
merged = pd.merge(merged, graphs, on='Species_ID')

def animate(img_path):
    forward = sorted(glob.glob(f'{img_path}*.png'))
    backward = sorted(forward, reverse=True)
    img, *imgs = [Image.open(f) for f in forward + backward]
    img.save(fp=f'{img_path}seq.gif', format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)

def remove_edges(G,rm_num=1):
    if G.number_of_edges() <= rm_num:
        return False
    G.remove_edges_from(random.choices(list(G.edges()),k=rm_num))
    return True
    
def remove_nodes(G,rm_num=1):
    if G.number_of_nodes() <= rm_num:
        return False
    G.remove_nodes_from(random.choices(list(G),k=rm_num))
    return True

def render(df,
           species_id,
           min_lcsg=.1,
           whole_graph=True,
           use_spring=False,
           animate_only=False,
           use_edges=False,
           ):
    img_path = f'{tmp_path}{species_id}'
    if use_spring:
        img_path += '/spring/'
    else:
        img_path += '/kamada_kawai/'
    if os.path.isdir(img_path):
        if animate_only:
            animate(img_path)
            return
        
        # clear any prior PNGs
        for f in glob.glob(f'{img_path}*.png'):
            os.remove(f)
    else:
        os.makedirs(img_path,exist_ok=True)
    
    G = nx.Graph(df.loc[species_id]['Matrix'])
    init_n = G.number_of_nodes()
    i = 0
    nn = []
    gg = []
    dd = []
    mm = []
    lcsg_mm = []
    diffs = []
    
    chaff = [node for node,degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(chaff)
    remove = remove_edges if use_edges else remove_nodes
    while G.number_of_nodes() > 0:
        comms = sorted(nx.connected_components(G), key=len)
        giant = G.subgraph(max(nx.connected_components(G), key=len, default=set()))
        if not whole_graph:
            G = G.subgraph(giant)
        
        n = G.number_of_nodes()
        nn.append(n)
        
        g = len(giant)
        gg.append(g)
        
        diffs.append(n-g)
        
        lcsg_ratio = g/n
        failure = lcsg_ratio < min_lcsg
        
        color_map = []
        if whole_graph:
            comms = nx_comm.label_propagation_communities(G)
            lcsg_comms = nx_comm.label_propagation_communities(giant)
            try:
                modularity = nx_comm.modularity(G, comms)
            except ZeroDivisionError:
                modularity = 0
            mm.append(modularity)
            try:
                lcsg_modularity = nx_comm.modularity(giant, lcsg_comms)
            except ZeroDivisionError:
                lcsg_modularity = 0
            lcsg_mm.append(lcsg_modularity)
            for node in G:
                if node in giant:
                    color_map.append('tab:blue')
                else:
                    color_map.append('tab:gray')
            
            # plot
            fig, axs = plt.subplots(
                2, 1,
                figsize=(10,12),
                gridspec_kw={'height_ratios': [3, 1]},
            )
            axs[0].set_title(
                f"{src_df['Compact Name'].loc[species_id]} "
                f'({n} nodes)')
            
            if use_spring:
                # a little hard to read but actually looks pretty good with chaff
                nx.draw(G, ax=axs[0], node_color=color_map, node_size=10, width=.6)
            else:
                # seems pretty good for the giant component but loses the chaff
                kawai = nx.kamada_kawai_layout(G)
                nx.draw(G, kawai, ax=axs[0], node_color=color_map, node_size=10, width=.6)
            
            # didn't get graphviz working with colors
            # nx.drawing.nx_agraph.graphviz_layout(G, node_color=color_map, node_size=10, width=.6)
            
            axs[1].plot(gg,color='tab:orange',label='giant')
            axs[1].plot(np.subtract(nn,gg),color='tab:blue',label='other')
            axs[1].set_ylabel('node counts')
            xtwin = axs[1].twinx()
            xtwin.plot(mm,'--',color='tab:red',label='total')
            xtwin.plot(lcsg_mm,color='tab:red',label='giant')
            xtwin.set_ylabel('modularity',color='red')
            xtwin.tick_params(axis='y', labelcolor='tab:red')
            xtwin.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            # legends
            box = axs[1].get_position()
            axs[1].set_position([box.x0, -box.y0 * .1,
                                 box.width, box.height * 0.9])
            axs[1].legend(
                loc='upper center', bbox_to_anchor=(.1, 1.2),
                fancybox=True, shadow=True, ncol=2,
                facecolor='white', framealpha=1)
            box = xtwin.get_position()
            xtwin.set_position([box.x0, -box.y0 * .3,
                                box.width, box.height * 0.9])
            xtwin.legend(
                loc='upper center', bbox_to_anchor=(.9, 1.2),
                fancybox=True, shadow=True, ncol=2,
                facecolor='white', framealpha=1)
            plt.subplots_adjust(hspace=.05)
            
            # output
            fig.tight_layout()
            fig.savefig(f'{img_path}{i:03d}.png')
            plt.close(fig)
            
        else:
            comms = sorted(nx_comm.label_propagation_communities(G), key=len)
            comm_ids = {n: comm_id for comm_id, comm in enumerate(comms) for n in comm}
            try:
                modularity = nx_comm.modularity(G, comms)
            except ZeroDivisionError:
                modularity = 0
            mm.append(modularity)
            for node in G:
                comm_id = comm_ids[node]
                if comm_id == 0:
                    color_map.append('tab:blue')
                elif comm_id == 1:
                    color_map.append('tab:green')
                elif comm_id == 2:
                    color_map.append('tab:black')
                else:
                    color_map.append('tab:gray')
            
            # plot
            plt.figure(figsize=(12,9))
            ax = plt.gca()
            ax.set_title(f"{src_df['Compact Name'].loc[species_id]} "
                          f'({n} nodes; modularity: {modularity:.3f}; {100 * lcsg_ratio:06.2f}% in Giant)')
            
            # layout
            if use_spring:
                # a little hard to read but actually looks pretty good with chaff
                nx.draw(G, node_color=color_map, node_size=10, width=.6)
            else:
                # seems pretty good for the giant component but loses the chaff
                kawai = nx.kamada_kawai_layout(G)
                nx.draw(G, kawai, node_color=color_map, node_size=10, width=.6)
                
            # output
            plt.tight_layout()
            plt.savefig(f'{img_path}{i:03d}.png')
            plt.clf()
            plt.close('all')
        
        if failure:
            print('network failure')
            break # network failure
        
        # remove and continue
        if not remove(G):
            print(f"can't remove any more {'edges' if use_edges else 'nodes'}!")
            break # can't remove any more!
        
        # remove chaff
        chaff = [node for node,degree in dict(G.degree()).items() if degree == 0]
        G.remove_nodes_from(chaff)
        i += 1
    
    # create gif and exit
    animate(img_path)

# X_all, y_all, merged_all = load_data()
# plt.hist(X_all[X_all['critical_threshold'] > 1000]['critical_threshold'])
# plt.yscale('log')
# plt.show()

# X_all['crit_per_node'] = X_all['critical_threshold']/np.log(X_all['total_nodes'])
# plt.hist(X_all['crit_per_node'])

# X_all.loc[9606]['critical_threshold']

# X['critical_range'] = X['max_critical'] - X['min_critical']
# get_extreme(X,'critical_range')
# for prefix in ['min', 'max', 'avg']:
#     val = X.loc[316407][f'{prefix}_critical']
#     print(f'{prefix}: {val}')


# tmp_df = src_df[src_df['Compact Name'].str.contains("Miyaz")]
# render(src_df,9606) # human!

# small = 492
# smallest_100 = X['avg_critical'].nsmallest(small)
# for i in range(small):
#     kth = smallest_100.index.values[i]
#     score = X.loc[kth]["avg_critical"]
#     name = src_df.loc[kth]["Compact Name"]
#     print(f'{i}: {score:} ({kth} - {name})')

# complement of subset
# others = src_df.loc[src_df['Taxonomy Level 2'] != 'Bacteria_Proteobacteria']

for species_id in merged.index.unique()[::-1]:
    for b in [True,False]:
        render(merged,species_id,use_spring=b)

# render(merged,658172) # probably somewhat interesting
# render(merged,883) # miyazaki
# render(merged,36870) # wigglesworth (spring)
# render(merged,71421) # should be quick
# render(merged,391600) # should be quickest, well it starts at ~8.5% giant
# render(merged,1028805) # cool name
# render(merged,469008) # nodes > cliques
# render(merged,638300) # another cool name
# render(merged,316407) # biggest range (of subset)
