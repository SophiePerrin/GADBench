import pandas as pd

url = "http://snap.stanford.edu/jodie/reddit.csv"

"""
# Lire seulement les 5 premières lignes du CSV distant
df = pd.read_csv(url, sep= , header=None, usecols=[0, 1, 2, 3], nrows=5)

print(df.head(1))

# Lire seulement les colonnes
df_columns = pd.read_csv(url, nrows=0)
print(df_columns.columns)
"""

import requests
import csv
from io import StringIO
import numpy as np


# Télécharger les 5 premières lignes
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    lines = []
    for i, line in enumerate(r.iter_lines(decode_unicode=True)):
        if i >= 5:
            break
        lines.append(line)

# Lire avec le module csv pour gérer correctement les virgules
for i, line in enumerate(lines):
    reader = csv.reader(StringIO(line))
    cols = next(reader)  # [:15]  # garder seulement les 4 premières colonnes
    print(f"Ligne {i}: {cols}")

#################################

# Lecture du CSV complet
# df = pd.read_csv(url, nrows=1000)
"""

# Relecture du CSV en forçant la 5e colonne en string
df = pd.read_csv(url, dtype={4: str}, nrows=1000)

# Séparer les 4 premières colonnes (métadonnées)
meta = df.iloc[:, :4].copy()

# Splitter la 5ᵉ colonne en colonnes numériques
col5_split = df.iloc[:, 4].astype(str).str.split(',', expand=True).astype(float)

# Renommer les colonnes de features proprement
col5_split.columns = [f"f{i}" for i in range(col5_split.shape[1])]

# Reconstituer le DataFrame final
final_df = pd.concat([meta, col5_split], axis=1)

print(final_df.head())

"""


# Chargement du CSV
# Relecture du CSV en forçant la 5ᵉ colonne en string
df = pd.read_csv(url, dtype={4: str}, nrows=1000)

import ast

# Supposons que les 4 premières colonnes sont tes meta-données
meta = df.iloc[:, :4]

# Transformer la colonne de features en chaînes puis en float
def ensure_string(x):
    if isinstance(x, list):
        return ','.join(map(str, x))
    else:
        return str(x)

df.iloc[:, 4] = df.iloc[:, 4].apply(ensure_string)

# 2. Split sur la virgule et enlever les espaces
features_str = df.iloc[:, 4].str.split(',', expand=True).apply(lambda col: col.str.strip())

# 3. Convertir en float
features = features_str.apply(pd.to_numeric, errors='coerce')

print("Meta shape :", meta.shape)
print("Features shape :", features.shape)


#########################
"""

# Créer une colonne "feature_count" = nombre d'éléments dans la liste de features
df["feature_count"] = df["comma_separated_list_of_features"].apply(lambda x: len(str(x).split(",")))

# Aperçu des 5 premières lignes
print(df[["user_id", "item_id", "feature_count"]].head())

# Vérifier les valeurs uniques du nombre de features
print("\nValeurs uniques du nombre de features :", df["feature_count"].unique())

# Vérifier si toutes les lignes ont le même nombre
print("\nNombre de features constant ? ->", df["feature_count"].nunique() == 1)

####################

# Séparer meta / features
meta = df.iloc[:, :4].copy()
features = df.iloc[:, 4:].to_numpy(dtype=np.float32)

print(meta.head())
print("Features shape :", features.shape)

# Nombre total de colonnes
total_cols = df.shape[1]

# Les 4 premières colonnes sont les métadonnées
nb_features = total_cols - 4
print("Nombre de features par ligne :", nb_features)

# Affichage des premières lignes pour vérifier
print(df.head())

"""