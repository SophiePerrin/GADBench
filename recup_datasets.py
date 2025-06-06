# import argparse
import utils as ut
import os
import numpy as np
import dgl
# import networkx as nx
import pickle
import warnings
warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))

# inutile tant que je fixe ma liste de datasets en la réduisant, comme ci-dessous, à ceux que je veux réellement
'''
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default=None)
args = parser.parse_args()
'''

datasets = ['reddit', 'weibo']

# inutile tant que je fixe ma liste de datasets en la réduisant, comme ci-dessus, à ceux que je veux réellement
'''
if args.datasets is not None:
    if '-' in args.datasets:
        st, ed = args.datasets.split('-')
        datasets = datasets[int(st):int(ed)+1]
    else:
        datasets = [datasets[int(t)] for t in args.datasets.split(',')]
    print('Evaluated Datasets: ', datasets)
'''

for dataset_name in datasets:
    data = ut.Dataset(dataset_name)
    # afficher des infos de base sur chaque graphe
    g = data.graph
    print(f"{dataset_name} → {g.num_nodes()} nodes, {g.num_edges()} edges")
    print(f"Node feature shape: {g.ndata['feature'].shape}")
    print(f"Label present: {'label' in g.ndata}")
    # sauvegarder les features ou graphes
    X = g.ndata['feature'].cpu().numpy()
    np.save(f"{dataset_name}_features.npy", X)
    # convertir en networkx
    nx_graph = dgl.to_networkx(g)
    with open(f"{dataset_name}.gpickle", "wb") as f:
        pickle.dump(nx_graph, f)
