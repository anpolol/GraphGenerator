import pickle
import numpy as np
import networkx as nx
import igraph as ig

name = "0.90.70.1315"
with open("../dataset/graph_" + str(name) + ".pickle", "rb") as f:
    G = pickle.load(f)

ad = np.mean(list(dict(G.degree()).values()))
number_connected_components = nx.number_connected_components(G)
num_nodes = G.number_of_nodes()

if number_connected_components == 1:
    iG = ig.Graph.from_networkx(G)
    avg_shortest_path = 0
    for shortest_path in iG.distances():
        for sp in shortest_path:
            avg_shortest_path += sp
    avg_s_p = avg_shortest_path / (num_nodes * num_nodes - num_nodes)

edges = G.edges()
labels = np.load("../dataset/graph_0.90.70.1315_labels.npy")
first_vector = []
second_vector = []

for edge in edges:
    id1, id2 = edge
    first_vector.append(labels[id1])
    second_vector.append(labels[id2])

print("assortativity", np.corrcoef(first_vector, second_vector))
print("connected components", number_connected_components)
print("number of nodes", num_nodes)
print("average shortest paths", avg_s_p)
print("clustering coefficient", np.mean(list(nx.clustering(G).values())))
print("average degree", ad)
