import dgl
import matplotlib.pyplot as plt
import torch
import os
import numpy as np

datasets = ['reddit', 'weibo']

graphs = {}  # Dictionnaire pour stocker les graphes
mat = {}  # Dictionnaire pour stocker les matrices numpy

# Boucle sur tous les datasets
for dataset_name in datasets:
    # ================================
    # 1. chargement des graphes et de leurs matrices d'adjacence A, des features de leurs noeuds x, 
    # et des labels de ces derniers, y.
    # ================================
    # Chemin pour entreposer les résultats qu'on veut pouvoir réutiliser 
    output_dir = f"/home/onyxia/work/GADBench/results/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)  # crée le dossier 

    arrays = np.load(f"{output_dir}/{dataset_name}_arrays.npz")
    mat[dataset_name] = {
        "A": arrays["A"],
        "X": arrays["x"],  
        "y": arrays["y"]
    }
    graph, _ = dgl.load_graphs(f"{output_dir}/{dataset_name}_graph.bin")
    graphs[dataset_name] = graph[0]

    g = graphs[dataset_name]       # Graphe DGL
    A = mat[dataset_name]["A"]     # Matrice d’adjacence numpy
    x = mat[dataset_name]["X"]     # Features numpy
    y = mat[dataset_name]["y"]     # Labels numpy

    # Supposons que g est ton graphe DGL
    # Calcul des degrés
    degrees_in = g.in_degrees()  # ou g.out_degrees() pour un graphe dirigé
    degrees_in = degrees_in.numpy()  # convertir en tableau NumPy pour matplotlib
    degrees_out = g.out_degrees()  # ou g.out_degrees() pour un graphe dirigé
    degrees_out = degrees_out.numpy() 

    # Bins : 0-10 par pas de 1, puis 10-50 par pas de 5, puis 50-5000 par pas de 100
    bins = np.concatenate([
        np.arange(0, 11, 1),      # 0,1,2,...,10
        np.arange(11, 51, 5),     # 11,16,21,...,50
        np.arange(51, degrees_in.max()+100, 100)
    ])
    # Affichage de l'histogramme
    plt.figure(figsize=(8, 5))
    plt.hist(degrees_in, bins=bins, color='skyblue', edgecolor='black')
    plt.title(f"Distribution des degrés des noeuds pour {dataset_name}")
    plt.xlabel("Degré")
    plt.ylabel("Nombre de noeuds")
    plt.grid(axis='y', alpha=0.75)
    output_path = f"{output_dir}/{dataset_name}_degree_hist.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{dataset_name} : nb noeuds = {g.num_nodes()}, min degré = {degrees_in.min()}, max degré = {degrees_in.max()}")
    print(graph[0].num_nodes())
    print(graph[0].num_edges())

    # Nombre de noeuds de degré 0
    degrees = degrees_in + degrees_out
    num_degree_zero = (degrees == 0).sum().item()

    print(f"Nombre de noeuds de degré 0 : {num_degree_zero}")
    # Nombre total de nœuds
    num_nodes = g.num_nodes()

    # Proportion de nœuds de degré 0
    prop_zero_degree = num_degree_zero / num_nodes

    print(f"Proportion de nœuds de degré 0 : {prop_zero_degree:.4f}")

    print(f"{dataset_name} -> degré min: {degrees.min()}, degré max: {degrees.max()}")
    unique, counts = np.unique(degrees, return_counts=True)
    print(dict(zip(unique, counts)))
