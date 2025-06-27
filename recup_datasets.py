
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

import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#############################################

# Fonctions pour l'étude des caractéristiques des graphes de données

#############################################


def analyser_arêtes(g, poids_key='count'):
    src, dst = g.edges()
    weights = g.edata[poids_key]

    num_edges = len(src)
    print(f"\n🔢 Nombre total d’arêtes : {num_edges}")

    # Auto-boucles et arêtes entre nœuds différents
    mask_self = src == dst
    mask_diff = src != dst

    weights_self = weights[mask_self]
    weights_diff = weights[mask_diff]

    print(f"🔁 Auto-boucles : {len(weights_self)} arêtes, poids min = {weights_self.min().item() if len(weights_self) > 0 else None}, max = {weights_self.max().item() if len(weights_self) > 0 else None}")
    print(f"🔗 Arêtes entre nœuds différents : {len(weights_diff)} arêtes, poids min = {weights_diff.min().item() if len(weights_diff) > 0 else None}, max = {weights_diff.max().item() if len(weights_diff) > 0 else None}")

    # Conversion CPU pour comparaison set
    src_np = src.cpu().numpy()
    dst_np = dst.cpu().numpy()
    weights_np = weights.cpu().numpy()

    edges_np = np.stack((src_np, dst_np), axis=1)
    edge_set = set(map(tuple, edges_np))

    sym_mask = np.array([(j, i) in edge_set for i, j in edges_np])
    asym_mask = ~sym_mask

    weights_sym = weights_np[sym_mask]
    weights_asym = weights_np[asym_mask]

    print(f"🔄 Arêtes avec arête inverse : {len(weights_sym)} arêtes, poids min = {weights_sym.min() if len(weights_sym) > 0 else None}, max = {weights_sym.max() if len(weights_sym) > 0 else None}")
    print(f"↪️ Arêtes sans arête inverse : {len(weights_asym)} arêtes, poids min = {weights_asym.min() if len(weights_asym) > 0 else None}, max = {weights_asym.max() if len(weights_asym) > 0 else None}")


def describe_dgl_graph(g, name, max_examples=5):
    print(f"📊 Résumé du graphe DGL du jeu de données {name}")
    print("-" * 40)
    print(f"Nombre de nœuds : {g.num_nodes()}")
    print(f"Nombre d'arêtes : {g.num_edges()}")
    print(g)

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

    print(f"Le graphe est-il homogène ? {g.is_homogeneous}")
    print(f"Le graphe est-il unibipartite ? {g.is_unibipartite}")
    print(f"Résultats de has nodes : {g.has_nodes}")

# Autre méthode pour matrice d'adjacence
    # Construire la matrice sparse pondérée
    # Si src et dst sont des numpy arrays, on les convertit en torch tensors
    src_tensor = torch.tensor(src) if not torch.is_tensor(src) else src
    dst_tensor = torch.tensor(dst) if not torch.is_tensor(dst) else dst
    if 'count' in g.edata:
        count = g.edata['count']         # tensor shape: [N, 1]
        count = count.squeeze()          # shape devient [N]
    
    num_nodes = g.num_nodes()

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

###
    src, dst = g.edges()
    num_edges = len(src)

# Arêtes entre nœuds différents
    mask_diff_nodes = src != dst
    num_diff_edges = mask_diff_nodes.sum().item()

    print(f"🔢 Nombre total d'arêtes : {num_edges}")
    print(f"🔗 Arêtes entre nœuds différents : {num_diff_edges}")
    print(f"🔁 Auto-boucles (i → i) : {num_edges - num_diff_edges}")

    weights = g.edata['count']
    min_w = weights.min().item()
    max_w = weights.max().item()

    print(f"📏 Plage des poids d’arêtes : min = {min_w}, max = {max_w}")

    # Créer un ensemble des paires (i, j) et (j, i)
    edge_set = set(zip(src.tolist(), dst.tolist()))
    reverse_set = set((j, i) for (i, j) in edge_set if i != j)

# Arêtes qui n'ont pas leur inverse
    asym_edges = reverse_set - edge_set
    print(len(asym_edges))
    if len(asym_edges) == 0:
        print("✅ Le graphe est symétrique : pour chaque arête i → j, il existe j → i.")
    else:
        print(f"⚠️ Le graphe est orienté : {len(asym_edges)} arêtes n’ont pas leur inverse.")
    analyser_arêtes(g)

#############################################

# Fonctions pour la réduction de dimension des features des noeuds des graphes (lorsque utile)
# et pour transformer un graphe réellement orienté en graphe non orienté (par la méthode de repondération des arcs en arêtes)

#############################################


def analyze_feature_redundancy(graph, variance_thresh=1e-6, corr_thresh=0.95, pca_variance=0.95):
    # 1. Extraire les features
    X = graph.ndata['feature'].numpy()

    # 2. Calculer la variance
    variances = X.var(axis=0)
    var_idx = np.where(variances >= variance_thresh)[0]       # indices à garder
    low_var_idx = np.where(variances < variance_thresh)[0]    # indices supprimés pour traçabilité
    print(f"{len(low_var_idx)} features ont une variance < {variance_thresh} : {low_var_idx.tolist()} — elles sont supprimées avant la PCA")

    # 3. Filtrer les colonnes à faible variance
    X_clean = X[:, var_idx]

    # 4. Mettre à jour les features du graphe
    graph.ndata['feature'] = torch.tensor(X_clean, dtype=torch.float32)

    # 5. Features très corrélées (calculé sur X d'origine, pas X_clean)
    corr_matrix = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(corr_matrix, 0)
    
    # Matrice de corrélation absolue
    abs_corr = np.abs(corr_matrix)

    # Indices de la partie triangulaire supérieure (hors diagonale)
    triu_indices = np.triu_indices_from(abs_corr, k=1)

    # Paires (i, j) avec leur valeur de corrélation
    pair_scores = [(i, j, abs_corr[i, j]) for i, j in zip(*triu_indices)]  # 🔄 remplacé redundant_pairs

    # Trier par corrélation décroissante
    pair_scores.sort(key=lambda x: x[2], reverse=True)  # 🔄 nouveau : trie toutes les paires par corrélation

    # Affichage
    print("Top 10 des paires de features les plus corrélées :")  # 🔄 message plus clair
    for i, j, score in pair_scores[:10]:                         # 🔄 on affiche les paires réellement les plus corrélées
        print(f"  Feature {i} ↔ Feature {j} (corr = {corr_matrix[i, j]:.2f})")

    # 6. PCA sur les features nettoyées
    pca = PCA(n_components=pca_variance)
    pca.fit(X_clean)
    print(f"PCA a réduit de {X_clean.shape[1]} à {pca.n_components_} dimensions (variance expliquée : {pca_variance})")

    # 7. Affichage des poids de la première composante
    comp_weights = np.abs(pca.components_[0])
    plt.bar(np.arange(len(comp_weights)), comp_weights)
    plt.title("Poids absolus des features dans la 1re composante principale")
    plt.xlabel("Feature index")
    plt.ylabel("Poids")
    plt.show()

    # 8. Remplacer les features du graphe par celles transformées par la PCA
    X_pca = pca.transform(X_clean)
    graph.ndata['feature'] = torch.tensor(X_pca, dtype=torch.float32)


    # 9. Retourner les résultats
    return {
        'features_supprimées_par_variance': low_var_idx.tolist(),  
        'top_corr_pairs': pair_scores[:10],                   
        'pca_model': pca,
        'graph_pca': graph 
    }


def make_weighted_undirected_with_node_features(g):
    # 1. Extraire les arêtes orientées
    src, dst = g.edges()

    # 2. Compter les relations (non orientées)
    edge_counts = {}
    for u, v in zip(src.tolist(), dst.tolist()):
        key = tuple(sorted([u, v]))
        edge_counts[key] = edge_counts.get(key, 0) + 1

    # 3. Préparer les arêtes et les poids
    new_src = []
    new_dst = []
    new_weights = []

    for (u, v), w in edge_counts.items():
        # Attribution du poids basé sur le type de relation
        if u == v:
            weight = 1.0  # boucle
        elif w == 1:
            weight = 0.5  # unidirectionnel
        else:
            weight = 1.0  # bidirectionnel

        # Arête u → v
        new_src.append(u)
        new_dst.append(v)
        new_weights.append(weight)

        # Arête v → u (sauf si boucle)
        if u != v:
            new_src.append(v)
            new_dst.append(u)
            new_weights.append(weight)

    # 4. Créer le graphe non orienté
    g_undir = dgl.graph((new_src, new_dst), num_nodes=g.num_nodes())

    # 5. Copier les features des nœuds
    for key in g.ndata:
        g_undir.ndata[key] = g.ndata[key].clone()

    # 6. Ajouter les poids aux arêtes
    g_undir.edata['count'] = torch.tensor(new_weights, dtype=torch.float)

    return g_undir


#############################################

# #### Etude des caractéristiques des datasets et création d'un dictionnaire pour pouvoir les manipuler séparément ensuite :

#############################################

datasets = ['reddit', 'weibo']

graphs = {}  # Dictionnaire pour stocker les graphes

# Boucle sur tous les datasets
for dataset_name in datasets:
    # Chargement du dataset avec GADBench
    data = ut.Dataset(name=dataset_name, prefix='GADBench/datasets/')
    g = data.graph  # Récupération du graphe DGL

    graphs[dataset_name] = g  # Stockage du graphe avec son nom

    describe_dgl_graph(g, dataset_name, 2)
    
graphs_modif = {}

#############################################

# Travail de transformation - adaptation du graphe de données reddit

# Ce graphe est symétrique, et chaque arête a une arête inverse. Dans DGL, tous les graphes sont de type orienté (il est impossible qu'ils ne l'y soient pas).
# C'est du à la spécificité de DGL : faire des graphes pour y faire tourner des GNN
# Notre graphe, étant symétrique, est donc déjà sous la bonne forme en pratique (il n'est orienté que parce que DGL lui attribue ce type)

# On a donc uniquement à se préoccuper des features des noeuds, en éliminant celles qui varient très faiblement entre les différents noeuds,
# puis en effectuant une ACP pour transformer nos features parfois très corrélées entre elles, en features orthogonales les unes aux autres, et moins nombreuses

#############################################

resultats_reddit = analyze_feature_redundancy(graphs['reddit'])

graphs_modif['reddit'] = resultats_reddit['graph_pca']

describe_dgl_graph(graphs_modif['reddit'], 'reddit_modif')
#############################################

# Travail de transformation - adaptation du graphe de données weibo

# weibo est réellement orienté, car de nombreux arcs n'ont pas d'arc retour : on crée ces arcs retour
# et on procède par repondération : 
# 0 pour l'absence totale d'arc entre deux noeuds
# 1 pour un arc A --> B et B --> A
# 0,5 pour un arc uniquement A --> B (sans présence d'arc retour dans le graphe d'origine)

#############################################

graphs_modif['weibo'] = make_weighted_undirected_with_node_features(graphs['weibo'])

describe_dgl_graph(graphs_modif['weibo'], 'weibo_modif')

resultats_weibo = analyze_feature_redundancy(graphs['weibo'], variance_thresh=1e-2, corr_thresh=0.95, pca_variance=0.99)

graphs_modif['weibo'] = resultats_weibo['graph_pca']

#############################################
for name, g in graphs_modif.items():
    print(name, g.num_nodes(), g.num_edges())

#############################################

# Export des noeuds+features (x), des labels (y) et de la matrice de similarité (issue de A remaniée) des graphes de données

# WARNING : SERA A ADAPTER AUX NOMS DES GRAPHES REMANIES

#############################################

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

    # Si les arêtes ont un attribut 'count' (poids des arêtes), on l’utilise ; sinon poids unitaire
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

    BUCKET = "sophieperrinlyon2"
    PREFIX = "albert/"

    fs = s3fs.S3FileSystem()

    for name, arr in [(f"x_{dataset_name}.npy", x), (f"y_{dataset_name}.npy", y), (f"A_{dataset_name}.npy", A)]:
        path = f"{BUCKET}/{PREFIX}{name}"
        with fs.open(path, "wb") as f:
            np.save(f, arr)
            print(f"  ✔ Uploaded {name}")

'''


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


