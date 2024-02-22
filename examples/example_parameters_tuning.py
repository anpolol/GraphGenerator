import pickle
from itertools import product
from typing import Any, AnyStr, Dict

import networkx as nx
import numpy as np
import sys 
sys.path.append(".")
from core.tuning_parameters import TuneParameters
from mpire import WorkerPool
from warnings import filterwarnings
filterwarnings("ignore")


def run_example_tuning(args: Dict[AnyStr, Any]) -> None:
    """
     Here we should set up graph characteristics which constructed graph should have in the following order:
     label assortativity, feature assortativity, clustering coefficient, average shortest-path length, average degree,
     last value should be equal 1 as it is responsible for number of connected components

    :param args: dict[str, any]: Dict of required graph characteristics of generated graph: label and feature
     assortativity, clustering coefficient, average degree and average length of shortest paths
    """
    try:
        label_assort = args["label_assort"]
    except:
        label_assort = [None]
    feature_assort = args["feature_assort"]
    cluster = args["cluster"]
    avg_shortest_paths = args["avg_shortest_paths"]
    avg_degree = args["avg_degree"]

    target_parameters = [
        x
        for x in product(
            label_assort, feature_assort, cluster, avg_shortest_paths, avg_degree, [1]
        )
    ]
    # print(target_parameters)

    main = TuneParameters(number_of_trials=500, characteristics_check=target_parameters)
    main.run()

    #посмотрим на пример сгенерированного графа, пусть это будет первый из списка
    #name = "".join(map(str, ([label_assort[0]] if label_assort[0] is not None else []) + [feature_assort[0], cluster[0], avg_shortest_paths[0], avg_degree[0]]))

    #with open("../dataset/graph_" + str(name) + ".pickle", "rb") as f:
    #    G = pickle.load(f)

    #ad = np.mean(list(dict(G.degree()).values()))
    #print("clustering coefficient", np.mean(list(nx.clustering(G).values())))
    #print("average degree", ad)



if __name__ == "__main__":
   
    #args["label_assort"] = [0.1]
    feature_assort = [x / 10 for x in range(10)]
    cluster = [x / 10 for x in range(10)]
    avg_shortest_paths = [x for x in range(1, 10)]
    avg_degree = [x for x in range(2, 10)]

    n_jobs = 10

    tasks = []

    print("generating tasks")
    for x in feature_assort:
        args = dict()
        args["feature_assort"] = [x]
        for y in cluster:
            args["cluster"] = [y]
            for z in avg_shortest_paths:
                args["avg_shortest_paths"] = [z]
                for w in avg_degree:
                    args["avg_degree"] = [w]
                    tasks.append([args.copy()])
    
    with WorkerPool(n_jobs=n_jobs) as pool:
        pool.map(run_example_tuning, tasks, progress_bar=True)
   