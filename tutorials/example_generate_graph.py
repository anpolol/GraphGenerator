import numpy as np

from core.generator import Main as Model

# to generate graph without tuning (this would be faster), we should first set up input parameters

number_of_nodes = 1000
min_degree = 1
max_degree = 500
mu = 0.5  # degree of connectivity of nodes with the same label, possible values are in range (0,1)
tau1 = 2.0  # power of degree distribution possible values are in range [2.0-3.0)
sigma_init = (
    1  # dispertion of each cluster attrbutes' possible values are in range [0.8,1.31)
)
dimension = 128  # dimension of attribute vector
num_classes = 5
eta = 0.2  # parameters of BTER possible values are in range [0.05-5]
rho = 0.6  # parameters of BTER possible values are in range [0.6,0.95]

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

G, _ = model.making_graph()
model.print_statistics(model.statistics())
