# import argparse
import utils as ut
import os
import numpy as np
import dgl
import networkx as nx
import pickle
import warnings
import torch
import numpy as np
warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))


def describe_dgl_graph(g, name, max_examples=5):
    print(f"📊 Résumé du graphe DGL du jeu de données {name}")
    print("-" * 40)
    print(f"Nombre de nœuds : {g.num_nodes()}")
    print(f"Nombre d'arêtes : {g.num_edges()}")

    print("\n🔑 Attributs des nœuds :")
    for key in g.ndata.keys():
        print(f" - {key}: shape = {g.ndata[key].shape}")
        if max_examples > 0:
            print(f"   Exemple : {g.ndata[key][:max_examples]}")

    print("\n🔑 Attributs des arêtes :")
    for key in g.edata.keys():
        print(f" - {key}: shape = {g.edata[key].shape}")
        if max_examples > 0:
            print(f"   Exemple : {g.edata[key][:max_examples]}")

    print("\n🧪 Masques (s’ils existent) :")
    for mask in ['train_mask', 'val_mask', 'test_mask']:
        if mask in g.ndata:
            print(f" - {mask} → {g.ndata[mask].sum().item()} nœuds")

    print("\n🔁 Quelques arêtes :")
    src, dst = g.edges()
    for i in range(min(max_examples, len(src))):
        print(f"   {src[i].item()} → {dst[i].item()}")



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

# Dossier où les fichiers de sortie seront enregistrés
output_dir = "export_graph_data"
os.makedirs(output_dir, exist_ok=True)  # Création s'il n'existe pas

# Boucle sur tous les datasets
for dataset_name in datasets:
    # Chargement du dataset avec GADBench
    data = ut.Dataset(name=dataset_name, prefix='GADBench/datasets/')
    g = data.graph  # Récupération du graphe DGL

    # Affichage de quelques infos de base
    print(f"{dataset_name} → {g.num_nodes()} nodes, {g.num_edges()} edges")
    print(f"Node feature shape: {g.ndata['feature'].shape}")
    print(f"Label present: {'label' in g.ndata}")
    describe_dgl_graph(g, dataset_name, 2)

    # ================================
    # 1. Extraction des features des nœuds
    # ================================
    # Passage des features en numpy pour export
    x = g.ndata['feature'].cpu().numpy()

    # ================================
    # 2. Extraction des étiquettes des nœuds
    # ================================
    if 'label' in g.ndata:
        # Si les labels sont présents, on les extrait
        y = g.ndata['label'].cpu().numpy()
    else:
        # Sinon, on utilise -1 pour indiquer l'absence d'étiquette
        y = np.full(g.num_nodes(), fill_value=-1)

    # ================================
    # 3. Création de la matrice de poids des arêtes
    # ================================
    num_nodes = g.num_nodes()

    # Initialisation d'une matrice (num_nodes x num_nodes) remplie de zéros
    similarities = np.zeros((num_nodes, num_nodes))

    # Récupération des arêtes (liste des paires source → destination)
    src, dst = g.edges()
    src = src.cpu().numpy()
    dst = dst.cpu().numpy()

    # Si les arêtes ont un attribut 'weight', on l’utilise ; sinon poids unitaire
    if 'weight' in g.edata:
        weights = g.edata['weight'].cpu().numpy()
    else:
        weights = np.ones(len(src))  # poids par défaut = 1

    # Remplissage de la matrice de similarités avec les poids
    for s, d, w in zip(src, dst, weights):
        similarities[s, d] = w
        similarities[d, s] = w  # si le graphe est non orienté (symétrique)

    

    # ================================
    # 4. Sauvegarde dans un fichier .npz compressé
    # ================================
    # Nettoyage du nom de fichier
    graph_name = dataset_name.replace("/", "_")

    # Enregistrement dans un fichier compressé contenant x, y, similarities
    np.savez_compressed(
        os.path.join(output_dir, f"{graph_name}.npz"),
        x=x,
        y=y,
        similarities=similarities
    )



'''  
    # sauvegarder les features ou graphes
    X = g.ndata['feature'].cpu().numpy()
    np.save(f"{dataset_name}_features.npy", X)
    # convertir en networkx
    nx_graph = dgl.to_networkx(g)
    with open(f"{dataset_name}.gpickle", "wb") as f:
        pickle.dump(nx_graph, f)

'''