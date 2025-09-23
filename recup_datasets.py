# import argparse

import decrire_graphes as dg
import modif_graphes as mg
import utils as ut
import os
import s3fs
import numpy as np
import dgl
import warnings
import torch
warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

#############################################

# #### Etude des caractéristiques des datasets et création d'un dictionnaire pour pouvoir les manipuler séparément ensuite :

#############################################

datasets = ['reddit', 'weibo']

graphs = {}  # Dictionnaire pour stocker les graphes

# Boucle sur tous les datasets
for dataset_name in datasets:
    # Chargement du dataset avec GADBench
    data = ut.Dataset(name=dataset_name, prefix='./datasets/')
    g = data.graph  # Récupération du graphe DGL

    graphs[dataset_name] = g  # Stockage du graphe avec son nom

    dg.describe_dgl_graph(g, dataset_name, 2)
    
graphs_modif = {}  # Dictionnaire pour stocker les graphes après les modifications faites ci-dessous

#############################################

# Travail de transformation - adaptation du graphe de données reddit

# Ce graphe est symétrique, et chaque arc a un arc inverse. Dans DGL, tous les graphes sont de type orienté (il est impossible qu'ils ne l'y soient pas).
# C'est du à la spécificité de DGL : faire des graphes pour y faire tourner des GNN.
# Notre graphe, étant symétrique, est donc déjà sous la bonne forme en pratique (il n'est orienté que parce que DGL lui attribue ce type).

# On a donc uniquement à se préoccuper des features des noeuds, en éliminant celles qui varient très faiblement entre les différents noeuds,
# puis en effectuant une ACP pour transformer nos features parfois très corrélées entre elles, en features orthogonales les unes aux autres, et moins nombreuses

#############################################

resultats_reddit = mg.analyze_feature_redundancy(graphs['reddit'], pca_variance=0.95)

graphs_modif['reddit'] = resultats_reddit['graph_pca']

dg.describe_dgl_graph(graphs_modif['reddit'], 'reddit_modif')

#############################################

# Travail de transformation - adaptation du graphe de données weibo

# weibo est réellement orienté, car de nombreux arcs n'ont pas d'arc retour : on crée ces arcs retour
# et on procède par repondération : 
# 0 pour l'absence totale d'arc entre deux noeuds
# 1 pour un arc A --> B et B --> A
# 0,5 pour un arc uniquement A --> B (sans présence d'arc retour dans le graphe d'origine)

#############################################

graphs_modif['weibo'] = mg.make_weighted_undirected_with_node_features(graphs['weibo'])

resultats_weibo = mg.analyze_feature_redundancy(graphs_modif['weibo'], variance_thresh=1e-2, corr_thresh=0.95, pca_variance=0.99)

graphs_modif['weibo'] = resultats_weibo['graph_pca']

dg.describe_dgl_graph(graphs_modif['weibo'], 'weibo_modif')

#############################################
for name, g in graphs_modif.items():
    print(name, g.num_nodes(), g.num_edges())

#############################################

# Export des noeuds+attributs (x), des labels (y), et de la matrice d'adjacence A 
# pour utilisation par HypHC

#############################################


def get_fs(bucket="projet-clustering-ano-graphe"):
    '''
    Cette fonction vérifie la présence des variables d’environnement AWS nécessaires
    puis crée et retourne un système de fichiers S3 (s3fs.S3FileSystem) 
    configuré avec ces identifiants et l’endpoint fourni.
    '''
       
    required = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_S3_ENDPOINT"]
    for var in required:
        if not os.environ.get(var):
            raise EnvironmentError(f"⚠️ Variable {var} manquante, vérifie ton script bash / Onyxia secrets")
    
    fs = s3fs.S3FileSystem(
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ.get("AWS_SESSION_TOKEN"),
        client_kwargs={
            "endpoint_url": os.environ["AWS_S3_ENDPOINT"],
            "region_name": "us-east-1"
        }
    )

    # Test de connexion sur le bucket du projet 
    try:
        fs.ls(bucket)
    except Exception as e:
        raise ConnectionError(f"⚠️ Impossible de se connecter au bucket {bucket} : {e}")
    
    return fs

#############################################

# Boucle sur tous les datasets :
# cette boucle prépare chaque graphe pour le(s) clustering en extrayant ses features et labels,
# en construisant sa matrice d’adjacence,en visualisant les matrices obtenues, 
# puis en exportant les éléments pour HypHC vers S3
# et en stockant ceux pour le clustering spectral en local.

#############################################

'''
En détails, pour chaque dataset, la boucle fait : 

1 - Extraction des features des nœuds : récupère les features feature de chaque nœud 
et les convertit en tableau NumPy.

2 - Extraction des étiquettes Y : récupère les labels des nœuds s’ils existent, 
sinon crée un vecteur rempli de None pour indiquer l’absence d’étiquettes.

3 - Construction de la matrice d’adjacence pondérée : initialise une matrice carrée A
(matrice d'adjacence du graphe) et la remplit avec les poids des arêtes (ou 1 si arête non pondérée).

4 - Export des données vers S3 : sauvegarde dans un bucket S3 les attributs x des noeuds, 
les labels y et la matrice d’adjacence A pour le dataset courant.
'''

for dataset_name, g in graphs_modif.items():
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
        # Sinon, on met None pour indiquer l'absence d'étiquette
        y = np.full(g.num_nodes(), fill_value=None)                      

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

    # ================================
    # 4. Sauvegarde en local des données utiles pour le clustering spectral
    # ================================
    # sauvegarde des matrices/features/labels
    np.savez(f"{dataset_name}_arrays.npz", A=A, x=x, y=y)

    # sauvegarde du graphe seul
    dgl.save_graphs(f"{dataset_name}_graph.bin", [graphs_modif["{dataset_name}"]])
    
    # ================================
    # 4. Sauvegarde en S3 sur le cloud du datalab INSEE des données utiles pour HypHC
    # ================================

    BUCKET = "projet-clustering-ano-graphe"
    PREFIX = "albert/"

    # fs = s3fs.S3FileSystem()

    endpoint_url = f"{os.environ['AWS_S3_ENDPOINT']}"
    fs = get_fs()

    for name, arr in [(f"x_{dataset_name}.npy", x), (f"y_{dataset_name}.npy", y), (f"A_{dataset_name}.npy", A)]:
        path = f"{BUCKET}/{PREFIX}{name}"
        with fs.open(path, "wb") as f:
            np.save(f, arr)
            print(f"  ✔ Uploaded {name}")
    print(dataset_name, g.num_nodes(), g.num_edges())