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

    model = Model(
        num_nodes=1000,
        max_d=max_degree,
        num_classes=num_classes,
        etta=eta,
        ro=rho,
        mu=mu,
        sigma_init=sigma_init,
        sigma_every=1,
        dim=dimension,
        power=tau1,
        min_d=min_degree,
    )

    G, _ = model.generate()
    assert model.statistics()["Number of nodes"] == 1000
    assert model.statistics()["Avg shortest path"] < 3.8
    assert (
        model.statistics()["Avg Degree"] < 4.6
        and model.statistics()["Avg Degree"] > 3.3
    )
    assert model.statistics()["Cluster"] < 0.2
    assert (
        model.statistics()["Label Assort"] < 0.7
        and model.statistics()["Label Assort"] > 0.5
    )
    assert (
        model.statistics()["Feature Assort"] < 0.55
        and model.statistics()["Feature Assort"] > 0.25
    )
