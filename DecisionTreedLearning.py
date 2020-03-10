import networkx as nx
from gensim.models import Word2Vec, keyedvectors
from node2vec import Node2Vec
import pandas as pd
dataFolder='C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Data\\'
df = pd.read_csv(dataFolder+'dataForModelling.csv')





# G=nx.Graph()
# cross_points = []
# tree_branches = [(0, 1), (1, 2)]
# G.add_nodes_from(cross_points)
# G.add_edges_from(tree_branches)
# G = G.to_undirected()
# node2vec= Node2Vec(G, dimensions=64, walk_length=3, num_walks=60, p=1,q=1)
# # Learn embeddings
# model = node2vec.fit(window=10, min_count=1,batch_words=4)