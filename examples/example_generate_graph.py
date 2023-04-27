from typing import Any, AnyStr, Dict

from core.attributed_generator import AttributedGenerator as Model
import numpy as np


def run_example_generate_graph(args: Dict[AnyStr, Any]) -> None:
    """
    To generate graph without tuning (this would be faster), we should first set up input parameters, then call Model
    :param args: dict[str, any]: Dict of input parameters for generator: min degree, mac dgree, mu, power for power
    law degree of degree distribution tau1, dispersion for attributes of cluster sigma_init, dimension of attribute
     vector, number of classes num_classes and two parameters of BTER eta and rho
    """
    min_degree = args["min_degree"]
    max_degree = args["max_degree"]
    mu = args["mu"]
    tau1 = args["tau1"]
    sigma_init = args["sigma_init"]
    dimension = args["dimension"]
    num_classes = args["num_classes"]
    eta = args["eta"]
    rho = args["rho"]

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
    model.print_statistics(model.statistics())


if __name__ == "__main__":
    np.random.seed(30)
    args = dict()
    args["number_of_nodes"] = 1000
    args["min_degree"] = 1
    args["max_degree"] = 500
    args[
        "mu"
    ] = 0.5  # degree of connectivity of nodes with the same label, possible values are in range (0,1)
    args[
        "tau1"
    ] = 2.0  # power of degree distribution possible values are in range [2.0-3.0)
    args[
        "sigma_init"
    ] = 1.0  # the variance of the attributes of each cluster possible values are in the range [0.8,1.31]
    args["dimension"] = 128  # dimension of attribute vector
    args["num_classes"] = 5
    args["eta"] = 0.2  # parameters of BTER possible values are in range [0.05-5]
    args["rho"] = 0.6  # parameters of BTER possible values are in range [0.6,0.95]
    run_example_generate_graph(args)
