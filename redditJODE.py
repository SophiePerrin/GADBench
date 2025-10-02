import pandas as pd

url = "http://snap.stanford.edu/jodie/reddit.csv"


# Lire seulement les 5 premières lignes du CSV distant
df = pd.read_csv(url, sep=r'\s+', header=None, usecols=[0, 1, 2, 3], nrows=5)

print(df.head(1))

# Lire seulement les colonnes
df_columns = pd.read_csv(url, nrows=0)
print(df_columns.columns)
