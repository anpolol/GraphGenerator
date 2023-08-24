import numpy as np

from core.attributed_generator import AttributedGenerator as Model

np.random.seed(30)


def test_graph_generator_statistics():
    min_degree = 1
    max_degree = 500
    mu = 0.5
    tau1 = 2.0
    sigma_init = 1
    dimension = 128
    num_classes = 5
    eta = 0.2
    rho = 0.6

    model = Model()
    params = {}
    params["num_nodes"] = 1000
    params["max_d"] = max_degree
    params["eta"] = eta
    params["rho"] = rho
    params["mu"] = mu
    params["sigma_init"] = sigma_init
    params["sigma_every"] = 1
    params["dim"] = dimension
    params["power"] = tau1
    params["min_d"] = min_degree
    params["num_classes"] = num_classes
    params["sizes"] = None
    params["manual"] = False
    params["d_manual"] = 0.75
    params["betta"] = 0.1

    G, _ = model.generate(params)
    stats = G.get_statistics(params)
    assert stats["Number of nodes"] == 1000
    assert stats["Avg shortest path"] < 3.8
    assert stats["Avg Degree"] < 4.6 and stats["Avg Degree"] > 3.3
    assert stats["Cluster"] < 0.2
    assert stats["Label Assort"] < 0.7 and stats["Label Assort"] > 0.5
    assert stats["Feature Assort"] < 0.55 and stats["Feature Assort"] > 0.25
