import os
import shutil
import json
from pathlib import Path

def duplicate_trial_folders(embeddings_dir='embeddings', datasets=['reddit', 'weibo'], num_trials=10):
    base_dir = Path(embeddings_dir)
    
    for dataset in datasets:
        dataset_dir = base_dir / dataset
        if not dataset_dir.exists():
            print(f"Dossier {dataset} non trouvé, passage au suivant")
            continue
            
        # Pour chaque modèle dans le dataset
        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            # Pour chaque mode (supervised/semi_supervised)
            for mode_dir in model_dir.iterdir():
                if not mode_dir.is_dir():
                    continue
                    
                # Chercher trial_0
                source_trial = mode_dir / 'trial_0'
                if not source_trial.exists():
                    print(f"Pas de trial_0 dans {mode_dir}, passage au suivant")
                    continue
                
                print(f"\nDuplication pour {dataset}/{model_dir.name}/{mode_dir.name}:")
                
                # Lire les métadonnées source
                meta_file = source_trial / 'run_metadata.json'
                if meta_file.exists():
                    with open(meta_file) as f:
                        source_meta = json.load(f)
                else:
                    print(f"Attention: pas de metadata dans {source_trial}")
                    source_meta = {}
                
                # Dupliquer vers trial_1 à trial_9
                for i in range(1, num_trials):
                    target_trial = mode_dir / f'trial_{i}'
                    
                    # Si le dossier existe déjà, on skip
                    if target_trial.exists():
                        print(f"  trial_{i} existe déjà, ignoré")
                        continue
                        
                    # Copier tous les fichiers
                    shutil.copytree(source_trial, target_trial)
                    print(f"  Copié trial_0 vers trial_{i}")
                    
                    # Mettre à jour metadata.json avec le bon trial_id
                    meta_file = target_trial / 'run_metadata.json'
                    if meta_file.exists():
                        with open(meta_file) as f:
                            meta = json.load(f)
                            meta['trial'] = i  # Mettre à jour trial_id
                            if 'seed' in meta:
                                meta['seed'] = meta['seed'] + (i * 10)  # Incrémenter le seed
                        with open(meta_file, 'w') as f:
                            json.dump(meta, f, indent=2)
                    
                print(f"Terminé: {num_trials-1} trials dupliqués")

if __name__ == '__main__':
    # Liste des datasets à traiter
    datasets = ['reddit', 'weibo']  # Ajoutez d'autres datasets si besoin
    
    print("Début de la duplication des trials...")
    duplicate_trial_folders(datasets=datasets)
    print("\nTerminé!")