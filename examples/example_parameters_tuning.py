import pickle
from itertools import product

import networkx as nx
import numpy as np

from core.tuning_parameters import TuneParameters

# Here we should set up graph characteristics which constructed graph should have in the following order:
# label assortativity, feature assortativity, clustering coefficient, average length of shortest paths, average degree,
# last value should be equal 1 as it is responsble for number of connected components

label_assort = [0.9]
feature_assort = [0.7]
cluster = [0.1]
avg_shortest_paths = [3]
avg_degree = [15]

target_parameters = [
    x
    for x in product(
        label_assort, feature_assort, cluster, avg_shortest_paths, avg_degree, [1]
    )
]

main = TuneParameters(number_of_trials=500, characteristics_check=target_parameters)
main.run()

name = "".join(
    list(
        map(
            lambda x: str(x),
            [
                label_assort[0],
                feature_assort[0],
                cluster[0],
                avg_shortest_paths[0],
                avg_degree[0],
            ],
        )
    )
)
with open("../dataset/graph_" + str(name) + ".pickle", "rb") as f:
    G = pickle.load(f)

ad = np.mean(list(dict(G.degree()).values()))
print("clustering coefficient", np.mean(list(nx.clustering(G).values())))
print("average degree", ad)
