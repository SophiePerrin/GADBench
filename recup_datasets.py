# import argparse
import utils as ut
import os
import s3fs
import numpy as np
import dgl
import networkx as nx
import pickle
import warnings
import torch
from torch import sparse
import numpy as np
warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))


def describe_dgl_graph(g, name, max_examples=5):
    print(f"📊 Résumé du graphe DGL du jeu de données {name}")
    print("-" * 40)
    print(f"Nombre de nœuds : {g.num_nodes()}")
    print(f"Nombre d'arêtes : {g.num_edges()}")

    nxg = g.to_networkx()
    print("Le graphe est-il orienté ?", nxg.is_directed())

    # Affichage de quelques infos de base
    print(f"Node feature shape: {g.ndata['feature'].shape}")
    print(f"Label present: {'label' in g.ndata}")

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
    '''
    print(g.etypes)
    for etype in g.canonical_etypes:
        print(f"{etype} features:", g.edges[etype].data.keys())
    print(g.edges[('_N', '_E', '_N')].data['count'])
    for etype in g.canonical_etypes:
        for key in g.edges[etype].data.keys():
            print(f"{etype} - {key}:")
            print(g.edges[etype].data[key])

    src, dst = g.edges(form='uv', etype=('_N', '_E', '_N'))
    counts = g.edges[('_N', '_E', '_N')].data['count']

    for i, (u, v, c) in enumerate(zip(src, dst, counts)):
        if i >= 100:
            break
        print(f"{u.item()} -> {v.item()} : count = {c.item()}")
    '''
    '''
    # 1. Récupérer les arêtes et la feature 'count'
    src, dst = g.edges(form='uv', etype=('_N', '_E', '_N'))
    counts = g.edges[('_N', '_E', '_N')].data['count']          # (num_edges, 1) ou (num_edges,)

# 2. Construire un masque booléen : True si count > 1
    mask = (counts.squeeze() > 1)   # .squeeze() enlève la dimension inutile si besoin

# 3. Appliquer le masque pour ne garder que les arêtes voulues
    src_keep = src[mask]
    dst_keep = dst[mask]
    count_keep = counts[mask]

# 4. Afficher les arêtes filtrées
    for u, v, c in zip(src_keep, dst_keep, count_keep):
        print(f"{u.item()} -> {v.item()} : count = {c.item()}")
    
    for ntype in g.ntypes:
        print(f"Attributs des nœuds de type {ntype} :")
        print(list(g.nodes[ntype].data.keys()))

    '''


datasets = ['reddit', 'weibo']


# Boucle sur tous les datasets
for dataset_name in datasets:
    # Chargement du dataset avec GADBench
    data = ut.Dataset(name=dataset_name, prefix='GADBench/datasets/')
    g = data.graph  # Récupération du graphe DGL

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
    A = np.zeros((num_nodes, num_nodes))

    # Récupération des arêtes (liste des paires source → destination)
    src, dst = g.edges()
    src = src.cpu().numpy()
    dst = dst.cpu().numpy()

    # Si les arêtes ont un attribut 'weight', on l’utilise ; sinon poids unitaire
    if 'count' in g.edata:
        count = g.edata['count']         # tensor shape: [N, 1]
        count = count.squeeze()          # shape devient [N]
        count = count.cpu().numpy()      # devient array([1., 2., ...])
    else:
        count = np.ones(len(src))  # poids par défaut = 1

    # Remplissage de la matrice de similarités avec les poids
    for s, d, w in zip(src, dst, count):
        A[s, d] = w
        # A[d, s] = w  # si le graphe est non orienté (symétrique)
    print(f"matrice d'adjacence : {A}")

    if np.allclose(A.T, A):
        print("matrice symétrique")
    else:
        print("matrice pas symétrique")

    print(f"Le graphe est-il homogène ? {g.is_homogeneous}")

# Autre méthode pour matrice d'adjacence
    # Construire la matrice sparse pondérée
    # Si src et dst sont des numpy arrays, on les convertit en torch tensors
    src_tensor = torch.tensor(src) if not torch.is_tensor(src) else src
    dst_tensor = torch.tensor(dst) if not torch.is_tensor(dst) else dst

    adj = torch.sparse_coo_tensor(
        indices=torch.stack([src_tensor, dst_tensor]),
        values=count,
        size=(num_nodes, num_nodes)
    )

# adj contient les poids des arêtes
    print(f"matrice d'adjacence autre méthode de calcul : {adj.to_dense()}")  # Affiche la matrice dense
    # Si adj est sparse
    dense_adj = adj.to_dense()

    # Convertir en numpy array
    np_adj = dense_adj.cpu().numpy()
    if np.allclose(np_adj.T,np_adj):
        print("matrice symétrique")
    else:
        print("matrice pas symétrique")





    BUCKET = "sophieperrinlyon2"
    PREFIX = "albert/"

    fs = s3fs.S3FileSystem()

    for name, arr in [(f"x_{dataset_name}.npy", x), (f"y_{dataset_name}.npy", y), (f"A_{dataset_name}.npy", A)]:
        path = f"{BUCKET}/{PREFIX}{name}"
        with fs.open(path, "wb") as f:
            np.save(f, arr)
            print(f"  ✔ Uploaded {name}")




'''
    # 1. Reconstruire le masque users/items
# Ici j'utilise le fait que seuls les users ont un label >= 0.
    labels = g.ndata['label']               # shape [10984]
    is_user = labels >= 0                   # True = user, False = item
    is_item = ~is_user

# 2. Comptage des nœuds de chaque type
    num_users = is_user.sum().item()
    num_items = is_item.sum().item()
    print(f"Users détectés : {num_users}")
    print(f"Items détectés : {num_items}")

# 3. Récupérer toutes les arêtes
    src, dst = g.edges()

# 4. Compter les 4 cas de figure
    mask_ui = is_user[src] & is_item[dst]    # user → item
    mask_iu = is_item[src] & is_user[dst]    # item → user
    mask_uu = is_user[src] & is_user[dst]    # user → user
    mask_ii = is_item[src] & is_item[dst]    # item → item

    counts = {
        "user→item" : mask_ui.sum().item(),
        "item→user" : mask_iu.sum().item(),
        "user→user" : mask_uu.sum().item(),
        "item→item" : mask_ii.sum().item(),
        "total edges": src.shape[0]
    }

    for k, v in counts.items():
        print(f"{k:10s} : {v}")

# 5. Pourcentage user→item
    pct_ui = counts["user→item"] / counts["total edges"] * 100
    print(f"\n% user→item  : {pct_ui:.2f}%")
'''
'''
import os
import s3fs
import numpy as np

#BUCKET_OUT = "sophieperrinlyon2"


BUCKET = "sophieperrinlyon2"
PREFIX = "albert/"

fs = s3fs.S3FileSystem()

x_reddit = np.load("x_reddit.npy")
y_reddit = np.load("y_reddit.npy")
A_reddit = np.load("A_reddit.npy")


for name, arr in [("x_reddit.npy", x_reddit), ("y_reddit.npy", y_reddit), ("A_reddit.npy", A_reddit)]:
    path = f"{BUCKET}/{PREFIX}{name}"
    with fs.open(path, "wb") as f:
        np.save(f, arr)
    print(f"  ✔ Uploaded {name}")
'''
'''
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
        A=A
    )

'''

'''  
    # sauvegarder les features ou graphes
    X = g.ndata['feature'].cpu().numpy()
    np.save(f"{dataset_name}_features.npy", X)
    # convertir en networkx
    nx_graph = dgl.to_networkx(g)
    with open(f"{dataset_name}.gpickle", "wb") as f:
        pickle.dump(nx_graph, f)

'''



'''  
    # sauvegarder les features ou graphes
    X = g.ndata['feature'].cpu().numpy()
    np.save(f"{dataset_name}_features.npy", X)
    # convertir en networkx
    nx_graph = dgl.to_networkx(g)
    with open(f"{dataset_name}.gpickle", "wb") as f:
        pickle.dump(nx_graph, f)

'''
