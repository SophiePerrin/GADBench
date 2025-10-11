import dgl
import utils as ut
import decrire_graphes as dg

datasets = ['reddit', 'weibo']

graphs = {}  # Dictionnaire pour stocker les graphes

# Boucle sur tous les datasets
for dataset_name in datasets:
    # Chargement du dataset avec GADBench
    data = ut.Dataset(name=dataset_name, prefix='./datasets/')
    g = data.graph  # Récupération du graphe DGL

    graphs[dataset_name] = g  # Stockage du graphe avec son nom

    labels = data.labels if hasattr(data, 'labels') else None
    dg.analyze_anomaly_grouping_dgl(g, labels)

# Extraire les labels (vérifie la clé : "labels" ou "label")
# labels = label_dict.get("labels", label_dict.get("label"))

# Appeler ta fonction d’analyse
    dg.analyze_anomaly_grouping_dgl(g, labels)
