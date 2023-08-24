from typing import Any, AnyStr, Dict

import numpy as np

from core.attributed_generator import AttributedGenerator as Model


def run_example_generate_graph(args: Dict[AnyStr, Any]) -> None:
    """
    To generate graph without tuning (this would be faster), we should first set up input parameters, then call Model
    :param args: dict[str, any]: Dict of input parameters for generator: min degree, mac dgree, mu, power for power
    law degree of degree distribution tau1, dispersion for attributes of cluster sigma_init, dimension of attribute
     vector, number of classes num_classes and two parameters of BTER eta and rho
    """
    model = Model()
    G, _ = model.generate(args)
    G.print_statistics(args)


if __name__ == "__main__":
    np.random.seed(30)
    args = dict()
    args["num_nodes"] = 1000
    args["min_d"] = 1
    args["max_d"] = 500
    args[
        "mu"
    ] = 0.5  # degree of connectivity of nodes with the same label, possible values are in range (0,1)
    args[
        "tau1"
    ] = 2.0  # power of degree distribution possible values are in range [2.0-3.0)
    args[
        "sigma_init"
    ] = 1.0  # the variance of the attributes of each cluster possible values are in the range [0.8,1.31]
    args["dim"] = 128  # dimension of attribute vector
    args["num_classes"] = 5
    args["eta"] = 0.2  # parameters of BTER possible values are in range [0.05-5]
    args["rho"] = 0.6  # parameters of BTER possible values are in range [0.6,0.95]
    args["power"] = 2.0
    args["sigma_every"] = 1
    args["sizes"] = None
    args["manual"] = False
    args["d_manual"] = 0.75
    args["betta"] = 0.1

    run_example_generate_graph(args)
