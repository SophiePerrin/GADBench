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

url = "http://snap.stanford.edu/jodie/reddit.csv"

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
    cols = next(reader)[:4]  # garder seulement les 4 premières colonnes
    print(f"Ligne {i}: {cols}")
