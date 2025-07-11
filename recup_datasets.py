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

# Fonctions pour l'étude des caractéristiques des graphes de données

#############################################


def analyser_aretes(g, poids_key='count'): # fonction utilisée tout à la fin de la fonction describe_dgl_graph()
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

    print(f"Le graphe est-il homogène ? {g.is_homogeneous}")
    print(f"Le graphe est-il unibipartite ? {g.is_unibipartite}")
    print(f"Résultats de has nodes : {g.has_nodes}")

# Matrice d'adjacence
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
    print(f"matrice d'adjacence : {adj.to_dense()}")  # Affiche la matrice dense
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
    analyser_aretes(g)

#############################################

# Fonctions pour la réduction de dimension des features des noeuds des graphes (lorsque utile)
# et pour transformer un graphe réellement orienté en graphe non orienté (par la méthode de repondération des arcs en arêtes)

#############################################


def analyze_feature_redundancy(graph, variance_thresh=1e-6, corr_thresh=0.95, pca_variance=0.95):
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
    data = ut.Dataset(name=dataset_name, prefix='./datasets/')
    g = data.graph  # Récupération du graphe DGL

    graphs[dataset_name] = g  # Stockage du graphe avec son nom

    describe_dgl_graph(g, dataset_name, 2)
    
graphs_modif = {} # Dictionnaire pour stocker les graphes après les modifications faites ci-dessous

#############################################

# Travail de transformation - adaptation du graphe de données reddit

# Ce graphe est symétrique, et chaque arc a un arc inverse. Dans DGL, tous les graphes sont de type orienté (il est impossible qu'ils ne l'y soient pas).
# C'est du à la spécificité de DGL : faire des graphes pour y faire tourner des GNN.
# Notre graphe, étant symétrique, est donc déjà sous la bonne forme en pratique (il n'est orienté que parce que DGL lui attribue ce type).

# On a donc uniquement à se préoccuper des features des noeuds, en éliminant celles qui varient très faiblement entre les différents noeuds,
# puis en effectuant une ACP pour transformer nos features parfois très corrélées entre elles, en features orthogonales les unes aux autres, et moins nombreuses

#############################################

resultats_reddit = analyze_feature_redundancy(graphs['reddit'], pca_variance=0.95)

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

resultats_weibo = analyze_feature_redundancy(graphs_modif['weibo'], variance_thresh=1e-2, corr_thresh=0.95, pca_variance=0.99)

graphs_modif['weibo'] = resultats_weibo['graph_pca']

describe_dgl_graph(graphs_modif['weibo'], 'weibo_modif')

#############################################
for name, g in graphs_modif.items():
    print(name, g.num_nodes(), g.num_edges())

#############################################

# Export des noeuds+features (x), des labels (y), calcul de la matrice de similarité (issue de A remaniée) des graphes de données
# pour le clustering spectral, et export de A pour utilisation par HypHC

#############################################

# fonctions utiles pour cette partie du programme


# Calcul de la similarité cosine entre features des noeuds par blocs (pour ne pas exploser la mémoire dispo)
def compute_cosine_similarity_matrix_blockwise(X, block_size=1000):
    N = X.shape[0]
    X = X.astype(np.float32)

    # Normalisation des vecteurs ligne de X
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / (norms + 1e-8)  # pour éviter la division par zéro

    # Matrice de sortie
    S = np.empty((N, N), dtype=np.float32)

    for i in range(0, N, block_size):
        Xi = X[i:min(i+block_size, N)]
        for j in range(0, N, block_size):
            Xj = X[j:min(j+block_size, N)]
            S_block = np.dot(Xi, Xj.T)
            S[i:i+Xi.shape[0], j:j+Xj.shape[0]] = S_block

    # transformation de la similarité cosine en une similarité comprise entre 0 et 1 
    S = 0.5 * (1.0 + S)
    S = np.clip(S, 0.0, 1.0)
    # Diagonale à 1.0 (au cas où il y aurait un flottement numérique)
    np.fill_diagonal(S, 1.0)
    return S

# Fonction proposée par chat GPT pour optimiser à la fois alpha et n_cluster dans le cas de clustering spectral non supervisé : 
def grid_search_alpha_k(A, Scosine, 
                        alphas=np.linspace(0, 1, 11), 
                        k_range=range(2, 11), 
                        metric='silhouette', 
                        n_jobs=-1, verbose=False):
    
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from joblib import Parallel, delayed
    import numpy as np

    assert metric in ['silhouette', 'calinski', 'davies'], "metric doit être 'silhouette', 'calinski' ou 'davies'"
    results = []
    best_result = None

    def evaluate(alpha, k):
        S = alpha * A + (1 - alpha) * Scosine
        try:
            model = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans')
            y_pred = model.fit_predict(S)

            if metric == 'silhouette':
                score = silhouette_score(1 - S, y_pred, metric='precomputed')
            elif metric == 'calinski':
                score = calinski_harabasz_score(S, y_pred)
            elif metric == 'davies':
                score = -davies_bouldin_score(S, y_pred)  # on inverse car plus petit = meilleur

            if verbose:
                print(f"[α={alpha:.2f}, k={k}] {metric} = {score:.3f}")

            return {'alpha': alpha, 'k': k, 'score': score, 'y_pred': y_pred, 'S': S}
        
        except Exception as e:
            if verbose:
                print(f"[α={alpha:.2f}, k={k}] Erreur : {e}")
            return None

    tasks = [(alpha, k) for alpha in alphas for k in k_range]

    all_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate)(alpha, k) for alpha, k in tasks
    )
    all_results = [r for r in all_results if r is not None]

    best_result = max(all_results, key=lambda r: r['score'])

    print(f"\n✅ Meilleur : alpha={best_result['alpha']:.2f}, k={best_result['k']}, score={best_result['score']:.3f}")

    return (
        all_results, 
        best_result['alpha'], 
        best_result['k'], 
        best_result['score'], 
        best_result['y_pred'], 
        best_result['S']  # 👈 matrice S du meilleur alpha
    )


'''    
# fonction pour optimiser alpha dans le cadre d'un clustering spectral supervisé
def optimize_alpha_spectral(A, Scosine, y, alphas=np.linspace(0, 1, 11), metric='ARI'):
    assert metric in ['ARI', 'NMI'], "metric doit être 'ARI' ou 'NMI'"
    n_clusters = len(np.unique(y[y != -1]))
    results = []
    best_result = None

    for alpha in alphas:
        S = alpha * A + (1 - alpha) * Scosine
        model = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
        y_pred = model.fit_predict(S)

        mask = y != -1
        ari = adjusted_rand_score(y[mask], y_pred[mask])
        nmi = normalized_mutual_info_score(y[mask], y_pred[mask])

        print(f"[α={alpha:.2f}] ARI={ari:.3f} | NMI={nmi:.3f}")
        result = {'alpha': alpha, 'ARI': ari, 'NMI': nmi, 'y_pred': y_pred}
        results.append(result)

        if best_result is None or result[metric] > best_result[metric]:
            best_result = result

    print(f"\n✅ Meilleur alpha (selon {metric}) : {best_result['alpha']:.2f} → {metric} = {best_result[metric]:.3f}")

    return results, best_result['alpha'], best_result[metric], best_result['y_pred']
'''

# Boucle sur tous les datasets
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
        # Sinon, on utilise -1 pour indiquer l'absence d'étiquette
        y = np.full(g.num_nodes(), fill_value=-1)                       # EST CE QUE CE TRUC LA EST UNE PRATIQUE OK ???

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
    # 4. Création de la matrice Scosine de similarité des features des noeuds (pour le clustering spectral ici 
    # - pour le clustering hyperbolique : ce sera fait dans HypHC)
    # ================================

    # j'ai x (numpy) la matrice des features des noeuds

    print("Min global :", x.min())
    print("Max global :", x.max())
    
    norms = np.linalg.norm(x, axis=1)
    print("Norme moyenne :", norms.mean())
    print("Norme max :", norms.max())

    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)

    norms = np.linalg.norm(x, axis=1)
    print("Nouvelle norme moyenne :", norms.mean())
    print("Nouvelle norme max :", norms.max())

    # Calcul de la similarité cosine entre features des noeuds par blocs (pour ne pas exploser la mémoire dispo)
    Scosine = compute_cosine_similarity_matrix_blockwise(x, block_size=1000)

    Scosine = np.exp(Scosine * 10)  # accentue les différences car sinon nos Scosine sont très "plates" (tout s'y ressemble !)

    # ================================
    # 5. Création de la matrice similarities, qui combine poids des arêtes et similarités entre features des noeuds,
    # avec un hyperparamètre alpha à optimiser.
    # On le fait ici uniquement pour le clustering spectral (pour le clustering hyperbolique, on va juste exporter A,
    # et on construira similarities dans HypHC directement)
    # ================================
    
    results, alpha_opt, k_opt, score_opt, y_opt, similarities = grid_search_alpha_k(
        A, Scosine,
        alphas=np.linspace(0, 1, 11),   # ou par ex. np.linspace(0.2, 0.8, 7)
        k_range=range(2, 11),           # k = nombre de clusters à tester
        metric='silhouette',           # ou 'calinski' ou 'davies'
        n_jobs=-1,                      # pour utiliser tous les cœurs CPU
        verbose=True                    # pour afficher l’avancement
        )

    ###################################

    # visualisation de A et Scosine

    ##################################

    def plot_similarity_matrices(A, Scosine, filename="matrices_similarite.png"):
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(A, cmap='viridis')
        plt.title("A")

        plt.subplot(1, 2, 2)
        plt.imshow(Scosine, cmap='viridis')
        plt.title("Scosine")

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()  # ferme proprement la figure pour éviter les fuites mémoire

    plot_similarity_matrices(A, Scosine, f"matrices_{dataset_name}.png")


    # ================================
    # 6. Sauvegarde en S3 sur le cloud du datalab INSEE
    # ================================

    BUCKET = "sophieperrinlyon2"
    PREFIX = "albert/"

    fs = s3fs.S3FileSystem()

    for name, arr in [(f"x_{dataset_name}.npy", x), (f"y_{dataset_name}.npy", y), (f"A_{dataset_name}.npy", A)]:
        path = f"{BUCKET}/{PREFIX}{name}"
        with fs.open(path, "wb") as f:
            np.save(f, arr)
            print(f"  ✔ Uploaded {name}")
    print(dataset_name, g.num_nodes(), g.num_edges())

