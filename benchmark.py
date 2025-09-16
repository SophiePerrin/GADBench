import argparse
import time
from utils import *
import pandas
import os
import s3fs         # ###
import numpy as np
import warnings
warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))


def set_seed(seed=3407):
    '''
    assure que les tirages aléatoires et certains calculs dépendent 
    toujours de la même graine (3407 par défaut), ce qui permet de reproduire
    exactement les mêmes résultats entre différents runs d’un script.
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def load_data_s3(name, dataset_name):               # ###
    '''
    Cette fonction charge un fichier NumPy (.npy) identifié par name et dataset_name.
        - Si le fichier est déjà présent en local → elle le lit directement.
        - Sinon → elle le télécharge depuis un bucket S3, puis le charge en mémoire.
        - Optionnellement, elle peut le mettre en cache local pour les prochaines utilisations.
    '''
    local_path = f"/tmp/{name}_{dataset_name}.npy"

    if os.path.exists(local_path):
        return np.load(local_path)

    # Paramètres S3
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]

    # Initialiser le système de fichiers S3
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

    # Spécifier le chemin dans le bucket
    BUCKET = "projet-clustering-ano-graphe"
    FILE_KEY_S3 = f"albert/{name}_{dataset_name}.npy"  # le chemin correct
    FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3

    # Charger le fichier .npy depuis S3
    with fs.open(FILE_PATH_S3, mode="rb") as f:
        array = np.load(f)

    # Vérification (optionnelle)
    print(array.shape)
    print(array.dtype)

    # Sauvegarde en local pour la prochaine fois
    # np.save(local_path, array)

    return array                                    # ###


parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--semi_supervised', type=int, default=0)
parser.add_argument('--inductive', type=int, default=0)
parser.add_argument('--models', type=str, default=None)
parser.add_argument('--datasets', type=str, default=None)
parser.add_argument('--use_clusters_hyp', action='store_true', help='Utiliser les embeddings hyperboliques en entrée du modèle')
parser.add_argument('--use_clusters_spectr', action='store_true', help='Utiliser les résultats du clustering spectral en entrée du modèle')
parser.add_argument('--use_clusters_tout', action='store_true', help='Utiliser les embeddings hyperboliques concaténés aux résultats du clustering spectral en entrée du modèle')


args = parser.parse_args()

columns = ['name']
new_row = {}
datasets = ['reddit', 'weibo', 'amazon', 'yelp', 'tfinance',
            'elliptic', 'tolokers', 'questions', 'dgraphfin', 'tsocial', 'hetero/amazon', 'hetero/yelp']
models = model_detector_dict.keys()

if args.datasets is not None:
    if '-' in args.datasets:
        st, ed = args.datasets.split('-')
        datasets = datasets[int(st):int(ed)+1]
    else:
        datasets = [datasets[int(t)] for t in args.datasets.split(',')]
    print('Evaluated Datasets: ', datasets)

if args.models is not None:
    models = args.models.split('-')
    print('Evaluated Baselines: ', models)

for dataset in datasets:
    for metric in ['AUROC mean', 'AUROC std', 'AUPRC mean', 'AUPRC std',
                   'RecK mean', 'RecK std', 'Time']:
        columns.append(dataset+'-'+metric)

results = pandas.DataFrame(columns=columns)
file_id = None
for model in models:
    model_result = {'name': model}
    for dataset_name in datasets:
        if model in ['CAREGNN', 'H2FD'] and 'hetero' not in dataset_name:
            continue
        time_cost = 0
        train_config = {
            'device': 'cuda',
            'epochs': 200,
            'patience': 50,
            'metric': 'AUPRC',
            'inductive': bool(args.inductive)
        }
        data = Dataset(dataset_name)

        # les embeddings sont dans un fichier .npy                          # ###

        clusters = None  # valeur par défaut

        match True:
            case _ if args.use_clusters_hyp:
                # Embeddings "leaves_emb" depuis S3
                clusters = load_data_s3("leaves_emb", dataset_name)

            case _ if args.use_clusters_spectr:
                # Résultats du clustering spectral
                clusters = y_spectral

            case _ if args.use_clusters_tout:
                # Concaténation des deux sources
                clusters_hyp = load_data_s3("leaves_emb", dataset_name)

                if clusters_hyp.shape[0] != y_spectral.shape[0]:
                    raise ValueError(
                        f"Incompatibilité de dimensions : "
                        f"leaves_emb a {clusters_hyp.shape[0]} lignes mais "
                        f"y_spectral en a {y_spectral.shape[0]}"
                    )

                clusters = np.concatenate([clusters_hyp, y_spectral], axis=1)

            case _:
                # Aucun cluster
                clusters = None

        # Affectation unique à data
        data.clusters = clusters
        cluster_dim = clusters.shape[1] if clusters is not None else 0

        # clusters = load_data_s3("leaves_emb", dataset_name)
        # data.clusters = clusters
        
        model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0, 'cluster_dim': cluster_dim}
        model_config['h_feats'] = 16
        # if model in ['GHRN', 'KNNGCN', 'AMNet', 'GT', 'GAT', 'GATv2', 'GATSep', 'PNA']:   # require more than 24G GPU memory
        # continue

        auc_list, pre_list, rec_list = [], [], []
        for t in range(args.trials):
            torch.cuda.empty_cache()
            print("Dataset {}, Model {}, Trial {}".format(dataset_name, model, t))
            data.split(args.semi_supervised, t)
            seed = seed_list[t]
            set_seed(seed)
            train_config['seed'] = seed
            detector = model_detector_dict[model](train_config, model_config, data)
            st = time.time()
            print(detector.model)
            test_score = detector.train()
            auc_list.append(test_score['AUROC']), pre_list.append(test_score['AUPRC']), rec_list.append(test_score['RecK'])
            ed = time.time()
            time_cost += ed - st
        del detector, data

        model_result[dataset_name+'-AUROC mean'] = np.mean(auc_list)
        model_result[dataset_name+'-AUROC std'] = np.std(auc_list)
        model_result[dataset_name+'-AUPRC mean'] = np.mean(pre_list)
        model_result[dataset_name+'-AUPRC std'] = np.std(pre_list)
        model_result[dataset_name+'-RecK mean'] = np.mean(rec_list)
        model_result[dataset_name+'-RecK std'] = np.std(rec_list)
        model_result[dataset_name+'-Time'] = time_cost/args.trials
    model_result = pandas.DataFrame(model_result, index=[0])
    results = pandas.concat([results, model_result])
    file_id = save_results(results, file_id)
    print(results)
