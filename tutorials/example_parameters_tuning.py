from itertools import product

import numpy as np

from core.tuning_parameters import TuneParameters

# Here we should set up graph characteristics which constructed graph should have in the following order:
# label assortativity, feature assortativity, clustering coefficient, average length of shortest paths, average degree,
# last value should be equal 1 as it is responsble for number of connected components

label_assort = [0.9, 0.1]
feature_assort = [0.8]
cluster = [0.1]
avg_shortest_paths = [3]
avg_degree = [10]
target_parameters = [
    x
    for x in product(
        label_assort, feature_assort, cluster, avg_shortest_paths, avg_degree, [1]
    )
]


main = TuneParameters(number_of_trials=500, characteristics_check=target_parameters)
main.run()
