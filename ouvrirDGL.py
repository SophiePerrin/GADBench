'''
import dgl
g = dgl.load_graphs("datasets/reddit")[0][0]
print(g)
print("Node data keys:", g.ndata.keys())
print("Edge data keys:", g.edata.keys())
# si edata contient 'time' ou 'weight' ou 'orig_edge', ça t'arrête la lecture
for k in g.edata.keys():
    print(k, g.edata[k][:10])
# métadonnées éventuelles (si graph sauvegardé avec meta)
# dgl.save_graphs stocke un dict meta? vérifier fichier .json à côté
'''

