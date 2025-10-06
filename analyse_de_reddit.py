import dgl
import decrire_graphes as dg

# Charger le graphe Reddit directement via DGL (aucun pickle)
graphs, label_dict = dgl.load_graphs("data/reddit/dgl_graph.bin")

# Récupérer le premier graphe
g = graphs[0]

# Extraire les labels (vérifie la clé : "labels" ou "label")
labels = label_dict.get("labels", label_dict.get("label"))

# Appeler ta fonction d’analyse
dg.analyze_anomaly_grouping_dgl(g, labels)
