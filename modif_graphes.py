import decrire_graphes as dg
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
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import laplacian
from numpy.linalg import eigvalsh
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from joblib import Parallel, delayed

#############################################

# Fonctions pour la réduction de dimension des features des noeuds des graphes (lorsque utile)
# et pour transformer un graphe réellement orienté en graphe non orienté (par la méthode de repondération des arcs en arêtes)

#############################################


def analyze_feature_redundancy(graph, variance_thresh=1e-6, corr_thresh=0.95, pca_variance=0.95):
    '''
    Cette fonction nettoie les attributs des nœuds (normalisation, suppression faible variance/redondance) 
    puis applique une PCA pour réduire la dimensionnalité et met à jour le graphe avec ces nouveaux attributs
    des noeuds.
    '''
    # 1. Extraire les features
    X = graph.ndata['feature'].numpy()

    # 1. vérifier leurs caractéristiques
    print("Min global :", X.min())
    print("Max global :", X.max())
    
    norms = np.linalg.norm(X, axis=1)
    print("Norme moyenne :", norms.mean())
    print("Norme max :", norms.max())
    print("Ecart-type :", X.std(axis=0))

    means = X.mean(axis=0)
    stds = X.std(axis=0)

    print("Moyenne min (= 0 si déjà centrée réduite):", means.min())
    print("Moyenne max (= 0 si déjà centrée réduite):", means.max())
    print("Écart-type min (= 1 si déjà centrée réduite) :", stds.min())
    print("Écart-type max (= 1 si déjà centrée réduite) :", stds.max())

    # Résultat : ni reddit ni weibo ne sont centrés réduits, alors qu'ils doivent l'être pour effectuer la PCA

    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    means = X.mean(axis=0)
    stds = X.std(axis=0)

    print("Moyenne min vérif (= 0 si centrée réduite):", means.min())
    print("Moyenne max vérif (= 0 si centrée réduite):", means.max())
    print("Écart-type min vérif (= 1 si centrée réduite) :", stds.min())
    print("Écart-type max vérif (= 1 si centrée réduite) :", stds.max())

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
    X_pca = pca.fit_transform(X_clean)
    print(f"PCA a réduit de {X_clean.shape[1]} à {pca.n_components_} dimensions (variance expliquée : {pca_variance})")

    means = X_pca.mean(axis=0)
    stds = X_pca.std(axis=0)

    print("Moyenne min vérif (= 0 si centrée réduite):", means.min())
    print("Moyenne max vérif (= 0 si centrée réduite):", means.max())
    print("Écart-type min vérif (= 1 si centrée réduite) :", stds.min())
    print("Écart-type max vérif (= 1 si centrée réduite) :", stds.max())

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
    '''
    Cette fonction convertit un graphe orienté en graphe non orienté pondéré.
    Pour cela, elle fusionne les arêtes (u, v) et (v, u), attribue un poids 
    1.0 si la relation est bidirectionnelle (ou boucle),
    0.5 si elle est unidirectionnelle, 
    puis recrée un graphe non orienté en recopiant les attributs des nœuds
    et en ajoutant les poids dans edata['count'].
    '''
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