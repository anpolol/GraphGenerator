import networkx as nx
import numpy as np
from scipy.stats import rv_discrete

from core.bter import BTER


def test_bter_degrees():
    min_d = 1
    max_d = 500
    probs = []

    sum = np.sum([1 / (pow(i, 2.0)) for i in range(min_d, max_d + 1)])

    for x in range(min_d, max_d + 1):
        probability = 1 / (pow(x, 2.0) * sum)
        probs.append(probability)
    probs = tuple(probs)

    rand_power_law = rv_discrete(min_d, max_d, values=(range(min_d, max_d + 1), probs))

    num_nodes = 1000
    degrees = np.sort(rand_power_law.rvs(size=num_nodes))
    params = dict()
    params["degrees"] = degrees
    params["etta"] = 0.5
    params["ro"] = 1
    params["d_manual"] = 0.75
    params["betta"] = 0.1

    model = BTER(params)
    G_model = model.build_graph()
    degrees_built = dict(G_model.degree()).values()
    assert G_model.number_of_nodes() == 1000
    assert min(degrees_built) == min_d
    assert (nx.number_connected_components(G_model)) == 1
