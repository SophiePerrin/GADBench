import numpy as np
import warnings
import torch
warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

#############################################

# Fonctions pour l'étude des caractéristiques des graphes de données

#############################################


def analyze_degres(g):
    """
    Calcule et affiche les statistiques descriptives des in-degree et out-degree d'un graphe DGL.
    
    Paramètres
    ----------
    g : dgl.DGLGraph
        Graphe à analyser.
    
    Retour
    ------
    dict
        Un dictionnaire contenant les stats pour in-degree et out-degree.
    """
    def describe_tensor(x):
        x = x.float()
        stats = {
            "moyenne": x.mean().item(),
            "écart-type": x.std(unbiased=False).item(),
            "min": x.min().item(),
            "Q1": torch.quantile(x, 0.25).item(),
            "médiane": torch.median(x).item(),
            "Q3": torch.quantile(x, 0.75).item(),
            "max": x.max().item(),
        }
        return stats

    in_deg = g.in_degrees()
    out_deg = g.out_degrees()

    stats_in = describe_tensor(in_deg)
    stats_out = describe_tensor(out_deg)

    # Affichage
    print("=== In-degree ===")
    print(f"Moyenne  : {stats_in['moyenne']:.2f}")
    print(f"Écart-type : {stats_in['écart-type']:.2f}")
    print(f"Min      : {stats_in['min']}")
    print(f"Q1       : {stats_in['Q1']}")
    print(f"Médiane  : {stats_in['médiane']}")
    print(f"Q3       : {stats_in['Q3']}")
    print(f"Max      : {stats_in['max']}\n")

    print("=== Out-degree ===")
    print(f"Moyenne  : {stats_out['moyenne']:.2f}")
    print(f"Écart-type : {stats_out['écart-type']:.2f}")
    print(f"Min      : {stats_out['min']}")
    print(f"Q1       : {stats_out['Q1']}")
    print(f"Médiane  : {stats_out['médiane']}")
    print(f"Q3       : {stats_out['Q3']}")
    print(f"Max      : {stats_out['max']}\n")

    return {"in-degree": stats_in, "out-degree": stats_out}


def analyser_aretes(g, poids_key='count'): # fonction utilisée tout à la fin de la fonction describe_dgl_graph()
    '''
    Cette fonction analyse la structure des arêtes d’un graphe DGL et affiche :

        -le nombre total d’arêtes,
        -la répartition entre auto-boucles et arêtes normales (avec leurs poids min/max),
        -la répartition entre arêtes ayant une arête inverse et celles qui n’en ont pas (avec leurs poids min/max).
    
    '''
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
    '''
    C'est une fonction d’exploration et de diagnostic d’un graphe DGL. Elle affiche un résumé complet du graphe, 
    en examinant ses nœuds, ses arêtes, ses attributs et sa structure.

    Voici ce qu'elle fait :

    🔹 1. Informations générales - la fonction :

        Affiche le nom du graphe, son nombre de nœuds et d’arêtes.
        Vérifie si le graphe est orienté (via NetworkX).
        Vérifie s’il est homogène et/ou unibipartite.

    🔹 2. Attributs des nœuds et arêtes - la fonction : 

        Montre la forme des tenseurs dans g.ndata (features, labels, masques, etc.), avec quelques exemples de valeurs.
        Montre aussi les attributs des arêtes (par ex. poids count).

    🔹 3. Masques de données : la fonction affiche combien de nœuds sont marqués comme train/val/test, si ces masques existent.

    🔹 4. Échantillons d’arêtes : la fonction liste quelques arêtes avec leur source et destination.

    🔹 5. Matrice d’adjacence - la fonction :

        Construit une matrice d’adjacence pondérée (avec count comme poids).
        Vérifie si la matrice est symétrique (→ graphe non orienté) ou non (→ graphe orienté).

    🔹 6. Analyse des arêtes : la fonction calcule combien d’arêtes sont :

        des auto-boucles (i → i),
        des arêtes entre nœuds différents (i → j avec i ≠ j).
        donne la plage de poids des arêtes (min et max).
        vérifie si chaque arête a son arête inverse (symétrie du graphe).
        appelle analyser_aretes(g) pour approfondir l’analyse des arêtes.

        7. Analyse sur les degrés des noeuds : degré moyen des noeuds du graphe, écart-type des
        degrés, degrés quartiles et médians.
    '''
    print(f"Résumé du graphe DGL du jeu de données {name}")
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
    analyze_degres(g)



######################################################################
# Script d'analyse topologique des anomalies 
def analyze_anomaly_grouping_dgl(g, labels):
    """
    Analyse la structure des anomalies dans un graphe DGL.
    
    Args:
        g: DGLGraph
        labels: Tensor 1D de taille (num_nodes,), contenant 0 (normal) ou 1 (anomalie)
    """
    import dgl
    import torch
    import numpy as np

    n_nodes = g.num_nodes()
    n_edges = g.num_edges()
    n_anom = (labels == 1).sum().item()
    print(f"Total de nœuds : {n_nodes:,}")
    print(f"Total d'arêtes : {n_edges:,}")
    print(f"Nœuds anormaux : {n_anom:,} ({100 * n_anom / n_nodes:.3f}%)")

    # --- 1. Trouver les voisins anormaux pour chaque nœud anormal
    src, dst = g.edges()
    src, dst = src.cpu(), dst.cpu()
    labels = labels.cpu()

    # Créer un masque des anomalies
    anom_mask = labels == 1
    anom_nodes = torch.nonzero(anom_mask, as_tuple=True)[0]

    # Calculer le nombre de voisins anormaux pour chaque nœud anormal
    # (on utilise DGL pour l'efficacité)
    has_anom_neighbor = 0
    for node in anom_nodes:
        neighbors = g.successors(node)
        if torch.any(anom_mask[neighbors]):
            has_anom_neighbor += 1

    prop_has_anom_neighbor = has_anom_neighbor / n_anom if n_anom > 0 else 0
    print(f"Proportion d'anomalies ayant ≥1 voisin anomal : {prop_has_anom_neighbor:.3f}")

    # --- 2. Sous-graphe induit par les anomalies
    sub_g = g.subgraph(anom_nodes)
    num_nodes_sub = sub_g.num_nodes()
    num_edges_sub = sub_g.num_edges()
    density_sub = num_edges_sub / (num_nodes_sub * (num_nodes_sub - 1)) if num_nodes_sub > 1 else 0
    density_global = n_edges / (n_nodes * (n_nodes - 1))

    print(f"Densité globale du graphe : {density_global:.6e}")
    print(f"Densité interne (sous-graphe anomalies) : {density_sub:.6e}")

    # --- 3. Taille des composantes connexes (sur le sous-graphe anomalies)
    # DGL ne fournit pas directement cette fonction, donc on utilise NetworkX ici, mais sur un petit graphe
    import networkx as nx
    nx_sub = sub_g.to_networkx().to_undirected()
    comp_sizes = [len(c) for c in nx.connected_components(nx_sub)]
    if comp_sizes:
        print(f"Nombre de composantes d'anomalies : {len(comp_sizes)}")
        print(f"Taille moyenne des composantes : {np.mean(comp_sizes):.2f}")
        print(f"Taille médiane : {np.median(comp_sizes):.2f}")
        print(f"Top 10 tailles : {sorted(comp_sizes, reverse=True)[:10]}")
    else:
        print("Aucune composante connectée détectée (aucune arête entre anomalies).")

    # --- 4. Interprétation rapide
    if density_sub > 5 * density_global:
        print("→ Les anomalies semblent former des sous-graphes denses (group anomalies).")
    elif density_sub > 1.5 * density_global:
        print("→ Les anomalies sont légèrement plus connectées entre elles que la moyenne.")
    else:
        print("→ Les anomalies paraissent dispersées ou isolées (point anomalies).")
