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
import numpy as np
warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))


def describe_dgl_graph(g, name, max_examples=5):
    print(f"📊 Résumé du graphe DGL du jeu de données {name}")
    print("-" * 40)
    print(f"Nombre de nœuds : {g.num_nodes()}")
    print(f"Nombre d'arêtes : {g.num_edges()}")
    
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


# inutile tant que je fixe ma liste de datasets en la réduisant, comme ci-dessous, à ceux que je veux réellement
'''
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default=None)
args = parser.parse_args()
'''

datasets = ['reddit', 'weibo']


# inutile tant que je fixe ma liste de datasets en la réduisant, comme ci-dessus, à ceux que je veux réellement
'''
if args.datasets is not None:
    if '-' in args.datasets:
        st, ed = args.datasets.split('-')
        datasets = datasets[int(st):int(ed)+1]
    else:
        datasets = [datasets[int(t)] for t in args.datasets.split(',')]
    print('Evaluated Datasets: ', datasets)
'''

# Dossier où les fichiers de sortie seront enregistrés
output_dir = "export_graph_data"
os.makedirs(output_dir, exist_ok=True)  # Création s'il n'existe pas

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

    # Si les arêtes ont un attribut 'weight', on l’utilise ; sinon poids unitaire
    if 'weight' in g.edata:
        weights = g.edata['weight'].cpu().numpy()
    else:
        weights = np.ones(len(src))  # poids par défaut = 1

    # Remplissage de la matrice de similarités avec les poids
    for s, d, w in zip(src, dst, weights):
        A[s, d] = w
        #A[d, s] = w  # si le graphe est non orienté (symétrique)
    '''
    # Sauvegarde de x, y et A
    np.save(f"x_{dataset_name}", x)
    np.save(f"y_{dataset_name}", y)
    np.save(f"A_{dataset_name}", A)
    '''
    
    BUCKET = "sophieperrinlyon2"
    PREFIX = "albert/"

    fs = s3fs.S3FileSystem()

    for name, arr in [(f"x_{dataset_name}", x), (f"y_{dataset_name}", y), (f"A_{dataset_name}", A)]:
        path = f"{BUCKET}/{PREFIX}{name}"
        with fs.open(path, "wb") as f:
            np.save(f, arr)
            print(f"  ✔ Uploaded {name}")
    
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

# %%


'''  
    # sauvegarder les features ou graphes
    X = g.ndata['feature'].cpu().numpy()
    np.save(f"{dataset_name}_features.npy", X)
    # convertir en networkx
    nx_graph = dgl.to_networkx(g)
    with open(f"{dataset_name}.gpickle", "wb") as f:
        pickle.dump(nx_graph, f)

'''