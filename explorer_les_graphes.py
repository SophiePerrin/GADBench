import dgl
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import utils as ut
import decrire_graphes as dg

datasets = ['reddit', 'weibo']

graphs = {}  # Dictionnaire pour stocker les graphes
mat = {}  # Dictionnaire pour stocker les matrices numpy
graphs_modif = {}  # Dictionnaire pour stocker les graphes après les modifications faites ci-dessous

# Boucle sur tous les datasets
for dataset_name in datasets:

    # Chargement du dataset avec GADBench
    data = ut.Dataset(name=dataset_name, prefix='./datasets/')
    g = data.graph  # Récupération du graphe DGL

    graphs[dataset_name] = g  # Stockage du graphe avec son nom

    #dg.describe_dgl_graph(g, dataset_name, 2)
    # ================================
    # 1. chargement des graphes et de leurs matrices d'adjacence A, des features de leurs noeuds x, 
    # et des labels de ces derniers, y.
    # ================================
    # Chemin où on a entreposé les résultats qu'on veut réutiliser 
    output_dir = f"/home/onyxia/work/GADBench/results/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)  # crée le dossier 

    arrays = np.load(f"{output_dir}/{dataset_name}_arrays.npz")
    mat[dataset_name] = {
        "A": arrays["A"],
        "X": arrays["x"],  
        "y": arrays["y"]
    }
    graphsm, _ = dgl.load_graphs(f"{output_dir}/{dataset_name}_graph.bin")
    graphs_modif[dataset_name] = graphsm[0]

    gm = graphs_modif[dataset_name]       # Graphe DGL
    A = mat[dataset_name]["A"]     # Matrice d’adjacence numpy
    x = mat[dataset_name]["X"]     # Features numpy
    y = mat[dataset_name]["y"]     # Labels numpy

    if np.allclose(A.T, A):
        print("matrice symétrique")
    else:
        print("matrice pas symétrique")

    # g est ton graphe DGL
    # Calcul des degrés
    degrees_in = g.in_degrees()  # ou g.out_degrees() pour un graphe dirigé
    degrees_in = degrees_in.numpy()  # convertir en tableau NumPy pour matplotlib
    degrees_out = g.out_degrees()  # ou g.out_degrees() pour un graphe dirigé
    degrees_out = degrees_out.numpy() 

    # Bins : 0-10 par pas de 1, puis 10-50 par pas de 5, puis 50-5000 par pas de 100
    bins = np.concatenate([
        np.arange(0, 50, 1), 
        np.arange(50, 501, 2),      # 0,1,2,...,10
        np.arange(501, 1501, 5),     # 11,16,21,...,50
        np.arange(1501, degrees_in.max()+100, 100)
    ])
    # Affichage des histogrammes
    plt.figure(figsize=(8, 5))
    plt.hist(degrees_in, bins=bins, color='skyblue', edgecolor='black')
    plt.title(f"Distribution des degrés entrants des noeuds pour {dataset_name}")
    plt.xlabel("Degré")
    plt.ylabel("Nombre de noeuds")
    plt.grid(axis='y', alpha=0.75)
    output_path = f"{output_dir}/{dataset_name}_degree_in_hist.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{dataset_name} : nb noeuds = {g.num_nodes()}, min degré = {degrees_in.min()}, max degré = {degrees_in.max()}")
    print(graphs[dataset_name].num_nodes())
    print(graphs[dataset_name].num_edges())

    plt.figure(figsize=(8, 5))
    plt.hist(degrees_out, bins=bins, color='skyblue', edgecolor='black')
    plt.title(f"Distribution des degrés sortants des noeuds pour {dataset_name}")
    plt.xlabel("Degré")
    plt.ylabel("Nombre de noeuds")
    plt.grid(axis='y', alpha=0.75)
    output_path = f"{output_dir}/{dataset_name}_degree_out_hist.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{dataset_name} : nb noeuds = {g.num_nodes()}, min degré = {degrees_out.min()}, max degré = {degrees_out.max()}")
    print(graphs[dataset_name].num_nodes())
    print(graphs[dataset_name].num_edges())

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

    # A est ta matrice d'adjacence (numpy array)
    num_total = A.size  # nombre total d'éléments
    num_below_half = (A < 0.5).sum()  # nombre d'éléments < 0.5
    prop_below_half = num_below_half / num_total  # proportion

    print(f"Nombre d'éléments < 0.5 : {num_below_half}")
    print(f"Proportion d'éléments < 0.5 : {prop_below_half:.4f}")

    num_equal_half = (A == 0.5).sum()  # nombre d'éléments = 0.5
    prop_equal_half = num_equal_half / num_total  # proportion

    print(f"Nombre d'éléments = 0.5 : {num_equal_half}")
    print(f"Proportion d'éléments = 0.5 : {prop_equal_half:.4f}")

    num_uppon_half = (A > 0.5).sum()  # nombre d'éléments > 0.5
    prop_uppon_half = num_uppon_half / num_total  # proportion

    print(f"Nombre d'éléments > 0.5 : {num_uppon_half}")
    print(f"Proportion d'éléments > 0.5 : {prop_uppon_half:.4f}")

    num_uppon_zeroun = (A > 0.1).sum()  # nombre d'éléments > 0.1
    prop_uppon_zeroun = num_uppon_zeroun / num_total  # proportion

    print(f"Nombre d'éléments > 0.1 : {num_uppon_zeroun}")
    print(f"Proportion d'éléments > 0.1 : {prop_uppon_zeroun:.4f}")

    num_zero = (A == 0.0).sum()  # nombre d'éléments = 0.0
    prop_zero = num_zero / num_total  # proportion

    print(f"Nombre d'éléments = 0.0 : {num_zero}")
    print(f"Proportion d'éléments = 0.0 : {prop_zero:.4f}")
