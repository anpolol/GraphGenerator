import pickle
from itertools import product

import networkx as nx
import numpy as np

from core.tuning_parameters_agm import TuneParameters

# Here we should set up graph characteristics which constructed graph should have in the following order:
# label assortativity, feature assortativity, clustering coefficient, average shortest-path length, average degree,
# last value should be equal 1 as it is responsible for number of connected components


feature_assort = [0.3]
cluster = [0.1]
avg_shortest_paths = [3]
avg_degree = [5]

target_parameters = [
    x
    for x in product(
         feature_assort, cluster, avg_shortest_paths, avg_degree, [1]
    )
]

main = TuneParameters(number_of_trials=500, characteristics_check=target_parameters)
main.run()

name = "".join(
    list(
        map(
            lambda x: str(x),
            [
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
