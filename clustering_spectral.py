# import argparse

import decrire_graphes as dg
import modif_graphes as mg
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

# fonctions utiles pour cette partie du programme

def compute_cosine_similarity_matrix_blockwise(X, block_size=1000, safety_factor=0.8, use_memmap=False, memmap_file='S.dat'):
    """
    Calcule la matrice de similarité cosinus de X par blocs, en limitant la mémoire utilisée.
    
    Args:
        X (np.ndarray): matrice (N, D)
        block_size (int): taille des blocs pour le calcul
        safety_factor (float): fraction maximale de RAM à utiliser
        use_memmap (bool): si True, la matrice S sera stockée sur disque
        memmap_file (str): chemin du fichier memmap si use_memmap=True
    
    Returns:
        np.ndarray ou np.memmap: matrice de similarité cosinus
    """
    import psutil
    
    N = X.shape[0]
    print(f"N = {N}, mémoire théorique pour matrice dense float32 = {N*N*4/1e9:.2f} Go")
    
    # RAM dispo
    mem = psutil.virtual_memory()
    gb_avail = mem.available / (1024**3)
    
    # Taille attendue
    gb_needed = N * N * 4 / (1024**3)
    print(f"   Mémoire nécessaire = {gb_needed:.2f} Go")
    print(f"   Mémoire disponible = {gb_avail:.2f} Go")
    
    if gb_needed > safety_factor * gb_avail:
        if not use_memmap:
            print("⚠️ Attention : mémoire RAM insuffisante pour stocker la matrice complète, activation automatique de memmap")
            use_memmap = True
    
    # S'assurer que X est float32 sans copier inutilement
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    
    # Normalisation in-place
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X /= (norms + 1e-8)
    
    # Matrice de sortie
    if use_memmap:
        S = np.memmap(memmap_file, dtype=np.float32, mode='w+', shape=(N, N))
    else:
        S = np.empty((N, N), dtype=np.float32)
    
    # Calcul par blocs
    for i in range(0, N, block_size):
        Xi = X[i:min(i+block_size, N)]
        for j in range(0, N, block_size):
            Xj = X[j:min(j+block_size, N)]
            S_block = np.dot(Xi, Xj.T)
            S[i:i+Xi.shape[0], j:j+Xj.shape[0]] = S_block
    
    # Transformation [0,1]
    S *= 0.5
    S += 0.5
    np.clip(S, 0.0, 1.0, out=S)
    np.fill_diagonal(S, 1.0)
    
    return S

    '''
import dgl
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def spectral_clustering_dgl(g, Scosine, 
                            alphas=np.linspace(0, 1, 5), 
                            k_range=range(2, 11), 
                            metric='silhouette',
                            device='cpu', verbose=False):
    """
    Clustering spectral non supervisé sur graphe DGL (massif),
    avec combinaison adjacency + similarité attributs.
    
    g : DGLGraph (non orienté, pondéré)
    Scosine : np.array ou sparse matrix [N x N] (similarité des features)
    alphas : liste de poids pour la combinaison
    k_range : liste de nombres de clusters à tester
    metric : 'silhouette', 'calinski', 'davies'
    """

    N = g.num_nodes()
    # Adjacence creuse depuis DGL
    A = g.adj_external(scipy_fmt="csr")
    results = []

    def evaluate(alpha, k):
        # Combinaison pondérée (sparse + dense possible)
        if sp.issparse(Scosine):
            S = alpha * A + (1 - alpha) * Scosine
        else:
            S = alpha * A + (1 - alpha) * sp.csr_matrix(Scosine)

        # Laplacien normalisé
        d = np.array(S.sum(1)).flatten()
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(d + 1e-10))
        L = sp.eye(N) - D_inv_sqrt @ S @ D_inv_sqrt

        # Approximation des k premiers vecteurs propres
        try:
            eigvals, eigvecs = eigsh(L, k=k, which='SM')  # small eigenvalues
            X = eigvecs

            # Clustering k-means
            y_pred = KMeans(n_clusters=k, n_init=10).fit_predict(X)

            # Score interne
            if metric == 'silhouette':
                score = silhouette_score(X, y_pred)
            elif metric == 'calinski':
                score = calinski_harabasz_score(X, y_pred)
            elif metric == 'davies':
                score = -davies_bouldin_score(X, y_pred)  # inversion
            else:
                raise ValueError("Metric inconnue")

            if verbose:
                print(f"[α={alpha:.2f}, k={k}] {metric} = {score:.3f}")

            return {'alpha': alpha, 'k': k, 'score': score, 'y_pred': y_pred}

        except Exception as e:
            if verbose:
                print(f"[α={alpha:.2f}, k={k}] Erreur: {e}")
            return None

    # Exploration (⚠️ coûteux si beaucoup d’α et de k)
    for alpha in alphas:
        for k in k_range:
            res = evaluate(alpha, k)
            if res is not None:
                results.append(res)

    # Choix du meilleur
    best = max(results, key=lambda r: r['score'])
    print(f"\n✅ Meilleur: α={best['alpha']:.2f}, k={best['k']}, score={best['score']:.3f}")

    return results, best
'''
'''
# Cette fonction fait désormais planter le serveur du ssp lab... 

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

'''    
# Cette fonction n'est pas adaptée du tout à ce qu'on veut faire : un clustering spectral NON supervisé...

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

# Boucle sur tous les datasets :
# cette boucle prépare chaque graphe pour le clustering en extrayant ses features et labels,
# en construisant sa matrice d’adjacence et sa similarité cosinus,
# en visualisant les matrices obtenues, puis en exportant le tout vers S3.

'''
En détails, pour chaque dataset, la boucle fait : 

4 - Calcul de la matrice de similarité cosinus Scosine entre features de nœuds : normalise les features, 
calcule leur similarité cosinus par blocs, puis applique une transformation exponentielle pour accentuer les différences.

5 - Préparation d’une matrice de similarité combinée : prévoit de combiner la matrice A 
et la similarité cosinus avec un hyperparamètre alpha (utilisée pour le clustering spectral, qui reste à faire).

Visualisation (optionnel, non exécuté ici): génère et sauvegarde une image comparant A et la matrice de similarité cosinus.

'''

for dataset_name, g in graphs_modif.items():
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

    # Calcul de la similarité cosine entre features des noeuds par blocs (pour ne pas exploser la mémoire disponible)
    Scosine = compute_cosine_similarity_matrix_blockwise(x, block_size=1000)

    Scosine = np.exp(Scosine * 10)  # accentue les différences car sinon nos Scosine sont très "plates" (tout s'y ressemble !)

    # ================================
    # 5. Création de la matrice similarities, qui combine poids des arêtes et similarités entre features des noeuds,
    # avec un hyperparamètre alpha à optimiser.
    # On le fait ici uniquement pour le clustering spectral (pour le clustering hyperbolique, on va juste exporter A,
    # et on construira similarities dans HypHC directement)
    # ================================
    
    # results, best = spectral_clustering_dgl(g, Scosine, alphas=np.linspace(0, 1, 5), k_range=2, metric='silhouette', device='cpu', verbose=False)
  
    '''
    # partie qui exécute le clustering spectral : 
    # fait planter l'environnement du data lab : à retravailler.

    results, alpha_opt, k_opt, score_opt, y_opt, similarities = grid_search_alpha_k(
        A, Scosine,
        alphas=np.linspace(0, 1, 11),   # ou par ex. np.linspace(0.2, 0.8, 7)
        k_range=range(2, 11),           # k = nombre de clusters à tester
        metric='silhouette',           # ou 'calinski' ou 'davies'
        n_jobs=-1,                      # pour utiliser tous les cœurs CPU
        verbose=True                    # pour afficher l’avancement
        )
    '''
    # fonction qui crée y_spectral (la sortie attendue du clustering spectral...) :
    # elle sera à remplacer par la fonction de clustering spectral (car pour l'instant,
    # elle crée simplement un vecteur aléatoire pour "remplir" y_spectral en attendant
    # qu'il soit rempli des vraies valeurs !)
    def make_random_clustering(g, k, seed=None):
        """
        Génère un vecteur d'affectations de clusters aléatoires pour un graphe DGL donné.
    
        Paramètres
        ----------
        g : le graphe DGL dont on veut simuler le clustering.
        k : le nombre de clusters.
        seed : int ou None - graine aléatoire pour la reproductibilité (optionnel).
    
        Retour
        ------
        y_random : np.ndarray de taille (num_nodes,) - vecteur d'affectations de clusters aléatoires.
        """

        if seed is not None:
            np.random.seed(seed)
    
        n = g.num_nodes()
        y_random = np.random.randint(low=0, high=k, size=n)

        print(y_random.shape)   # (n,)
        print(y_random[:20])    # aperçu des 20 premières affectations
        return y_random

    y_spectral = make_random_clustering(g, 5, seed=43)
    
   
    # Chemin pour entreposer les résultats qu'on veut pouvoir réutiliser 
    output_dir = f"/home/onyxia/work/GADBench/results/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)  # crée le dossier 

    # Sauvegarde de y_spectral pour pouvoir le réutiliser dans benchmark.py
    np.save(f"{output_dir}/y_spectral.npy", y_spectral)
    print(f"y_spectral sauvegardé dans {output_dir}/y_spectral.npy")

'''
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
    '''
