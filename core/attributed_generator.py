import collections
from collections import deque
from typing import Any, AnyStr, Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
from community import community_louvain
from scipy.stats import rv_discrete

from core.bter import BTER
from core.graph import Graph
from core.utils import pk, sum


class AttributedGenerator:
    def __init__(
        self,
    ) -> None:
        """
        The generator for graphs with controllable graph characteristics based on BTER model
        """

        super().__init__()

    def making_degree_dist(
        self, min_d: int, max_d: int, num_nodes: int, mu: float, power: float
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Build three lists of degrees of nodes: overall, inside group and outside.

        :param min_d: (int): Degree value of the node with the minimum degree
        :param max_d: (int): Degree value of the node with the maximum degree
        :param num_nodes: (int): Number of nodes in the required graph
        :param mu: (float): required label assortativity
        :param power: (float): power of degree distribution
        :return: (([int],[int],[int])): Lists of degree of nodes, degrees of nodes inside their respective classes and
        degrees outside their own class
        """
        rand_power_law = rv_discrete(
            min_d, max_d, values=(range(min_d, max_d + 1), pk(min_d, max_d, power))
        )

        degrees = np.sort(rand_power_law.rvs(size=num_nodes))
        degrees_out = []
        degrees_in = (np.round(degrees * mu)).astype(np.int32)

        for j, deg in enumerate(degrees):
            degrees_out.append(deg - degrees_in[j])

        counter = collections.Counter(degrees)
        k = 0

        for i in range(1, int(np.ceil(1 / mu))):
            if i in counter:
                ones = 0
                for deg in degrees:
                    if deg == i:
                        ones += 1
                prob = torch.bernoulli(torch.ones(ones) * mu).numpy()
                degrees_in[k: k + ones] = prob * i
                degrees_out[k: k + ones] = (np.ones(ones) - prob) * i
                k = k + ones

        return degrees, degrees_in, degrees_out

    def making_clusters(
        self, num_classes: int, degrees_in: List[int]
    ) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
        """
        Build dictinary mappings of labels to degrees and nodes to labels

        :param num_classes: (int): Number of classes
        :param degrees_in: ([int]): List of degrees inside respective group
        :return: ({int: int}, {int: int}, {int: int})
        """
        # uniform selection
        labels_degrees = {}
        mapping = {}
        clusters = {}

        degrees_to_cluster = sorted(degrees_in)
        nodes = np.argsort(degrees_in)
        for j, (node, degree) in enumerate(list(zip(nodes, degrees_to_cluster))):
            if j % num_classes not in labels_degrees:
                labels_degrees[j % num_classes] = []

            labels_degrees[j % num_classes].append(degree)
            clusters[node] = j % num_classes

            if j % num_classes not in mapping:
                mapping[j % num_classes] = {}
                mapping[j % num_classes][0] = node
            else:
                mapping[j % num_classes][
                    max(mapping[j % num_classes].keys()) + 1
                ] = node

        return labels_degrees, mapping, clusters  # clusters - label for each vertex

    def making_clusters_with_sizes(
        self,
        num_nodes: int,
        num_classes: int,
        degrees_in: List[int],
        size_ratio: List[float],
    ) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
        """
        Make labels for nodes forcing the ratios of the sizes of classes according to size_ration list

        :param num_nodes: (int): Number of nodes in required graph
        :param num_classes: (int): Number of classes
        :param degrees_in: ([int]): List of degrees inside respective group
        :param size_ratio: ([float]): List of ratios of the sizes of classes of nodes
        :return: ({int: int}, {int: int}, {int: int}): Tuple of three dicts: first dict contains node degrees for each
        label, second -- mapping if index from old (in general list) to new (from local list) for each label, third --
        labels for each new index in local lists
        """
        degrees_in = deque(degrees_in)

        labels_degrees = {}
        mapping = {}
        clusters = {}

        size_ratio[::-1].sort()
        sizes = np.round(np.array(size_ratio) * (len(degrees_in) / sum(size_ratio)))
        if sum(sizes) <= num_nodes - 1:
            sizes[0] += num_nodes - sum(sizes)

        for k in range(num_classes):
            labels_degrees[k] = deque([])
            mapping[k] = {}

        list_of_classes = deque(range(num_classes))  # contains class numbers
        first_idx = (
            0  # first OLD index. Needed for mapping from the new index to the old one
        )
        last_idx = len(degrees_in) - 1  # last OLD index

        while len(list_of_classes) != 0:
            list_classes = list(list_of_classes)
            for l in list_classes:
                if len(labels_degrees[l]) < sizes[l] - 1:
                    mid = int(len(labels_degrees[l]) / 2)
                    labels_degrees[l].insert(mid, degrees_in.popleft())
                    labels_degrees[l].insert(mid + 1, degrees_in.pop())

                    clusters[first_idx] = l
                    clusters[last_idx] = l

                    first_new_idx = mid
                    last_new_idx = sizes[l] - 1 - mid

                    mapping[l][first_new_idx] = first_idx
                    mapping[l][last_new_idx] = last_idx

                    first_idx += 1
                    last_idx -= 1

                elif len(labels_degrees[l]) == sizes[l] - 1:
                    mid = int(len(labels_degrees[l]) / 2)
                    labels_degrees[l].insert(mid, degrees_in.popleft())

                    clusters[first_idx] = l

                    mapping[l][mid] = first_idx
                    first_idx += 1

                    list_of_classes.remove(l)

                else:
                    list_of_classes.remove(l)

        return labels_degrees, mapping, clusters

    def generate(self, params: Dict[AnyStr, Any]) -> Tuple[nx.Graph, Dict[int, int]]:
        """
        Generate graph, main function

         :param params: (dic): Dict of parameters for generator.
         General parameters:
         max_d: (int): Degree value of the node with the maximum degree
         num_classes: (int): Number of classes/labels in graph
         etta: (float): The hyperparameter for BTER
         ro: (float): The hyperparameter for BTER
         mu: (float): Required label assortativity
         dim: (float): Dimension of attribute vector
         power: (float): The power of the degree distribution (default: 2)
         sizes: ([int]): Sizes of classes, if None, the size will be approximately the same (default: None)
         manual: (bool): Flag identifying the way of connecting edges between classes -- mannually or using
          GeneratorNoAttr
         min_d: (int): Degree value of the node with the minimum degree
         sigma_init: (float): Variance of the normal distribution from which the attribute vectors of each class are
         taken separately
         sigma_every: (float): Variance of noise added to the vector of attributes pf each class for every node

        and attributes for GeneratorNoAttr
        d_manual: (float): The hyperparameter of BTER model (default: 0.75)
        betta: (float): The hyperparameter of BTER model (default: 0.1)

        :return: ((networkx.Graph, {int: int})): Graph if type networkx.Graph and mapping nodes to labels
        """
        num_nodes = params["num_nodes"]
        max_d = int(params["max_d"])
        min_d = int(params["min_d"])

        num_classes = params["num_classes"]
        etta = params["eta"]
        ro = params["rho"]
        mu = params["mu"]
        power = params["power"]
        sigma_init = params["sigma_init"]
        sigma_every = params["sigma_every"]
        dim = params["dim"]
        class_distr = params["sizes"]
        manual = params["manual"]
        d_manual = params["d_manual"]
        betta = params["betta"]

        degrees, degrees_in, degrees_out = self.making_degree_dist(
            min_d, max_d, num_nodes, mu, power
        )
        if class_distr is not None:
            labels_degrees, mapping, clusters = self.making_clusters_with_sizes(
                num_classes, degrees_in, class_distr
            )
        else:
            labels_degrees, mapping, clusters = self.making_clusters(
                num_classes, degrees_in
            )

        graph = Graph()
        for j in range(num_nodes):
            graph.add_node(j, label=clusters[j])

        # first collect edges with other classes
        if manual is True:
            G_out = self.manual_out_degree(degrees_out, clusters)
            graph.add_edges_from(G_out.edges())
        else:
            G_out, mapping_new2_to_new = self.bter_model_edges(
                degrees_out, etta, ro, d_manual, betta
            )
            for edge in G_out.edges():
                graph.add_edge(
                    mapping_new2_to_new[edge[0]], mapping_new2_to_new[edge[1]]
                )

        # now inside the classes we collect edges
        for label in labels_degrees:
            degrees_in = labels_degrees[label]

            G_in, mapping_new2_to_new = self.bter_model_edges(
                degrees_in, etta, ro, d_manual, betta
            )

            for edge in G_in.edges():
                graph.add_edge(
                    mapping[label][mapping_new2_to_new[edge[0]]],
                    mapping[label][mapping_new2_to_new[edge[1]]],
                )

        graph = self.generate_attributes(graph, sigma_init, sigma_every, dim)
        return graph, clusters

    def bter_model_edges(
        self, degrees: List[int], eta: float, rho: float, d_manual: float, betta: float
    ) -> Tuple[nx.Graph, Dict[int, int]]:
        """
        Add edges to nodes with degrees

        :param degrees: ([int]): Degrees of nodes
        :param eta: (float): The hyperparameter for BTER model
        :param rho: (float): The hyperparameter for BTER model
        :param d_manual: (float): The hyperparameter of BTER model (default: 0.75)
        :param betta: (float): The hyperparameter of BTER model (default: 0.1)
        :return: ((networkx.Graph, {int: int})): Graph with added edges and mapping of indices into degreees
        """
        w = 0
        mapping_new2_to_new = {}
        degrees_new = []

        for e, deg in enumerate(degrees):
            if deg != 0:
                mapping_new2_to_new[w] = e
                w += 1
                degrees_new.append(deg)
        params_for_noattr_generator = dict()
        params_for_noattr_generator["degrees"] = degrees_new
        params_for_noattr_generator["eta"] = eta
        params_for_noattr_generator["rho"] = rho
        params_for_noattr_generator["d_manual"] = d_manual

        params_for_noattr_generator["betta"] = betta
        model_degrees = BTER()
        G_model = model_degrees.build_subgraph(params_for_noattr_generator)

        return G_model, mapping_new2_to_new

    def manual_out_degree(
        self, degrees_out: List[int], clusters: Dict[int, int]
    ) -> nx.Graph:
        """
        Calculate edges between differenet classes in manual regime

        :param degrees_out: ([int]): List of degrees of nodes to different classes
        :param clusters: ({int: int}): Mapping of nodes into labels
        :return: (networkx.Graph): Constructed graph on out degrees of type networkx.Graph
        """
        G_model = Graph()

        while sum(degrees_out) > 0:
            j = 0
            while j < len(degrees_out):
                if degrees_out[j] > 0:
                    last = len(degrees_out) - 1
                    if (clusters[last] == clusters[j]) or (
                        (j, last) in G_model.edges()
                    ):
                        while (clusters[last] == clusters[j]) or (
                            (j, last) in G_model.edges()
                        ):
                            last -= 1
                    if j < last:
                        if degrees_out[last] > 0:
                            G_model.add_edge(j, last)
                            degrees_out[last] -= 1
                            degrees_out[j] -= 1
                        else:
                            degrees_out = degrees_out[:last]
                            last = len(degrees_out) - 1
                            if clusters[last] == clusters[j]:
                                while clusters[last] == clusters[j]:
                                    last -= 1
                            G_model.add_edge(j, last)
                            degrees_out[last] -= 1
                            degrees_out[j] -= 1
                j += 1

        return G_model

    def generate_attributes(
        self, graph: nx.Graph, sigma_init: float, sigma_every: float, m: int
    ) -> None:
        """
        Add attributes to nodes in the Graph

        :param graph: (nx.Graph): input graph to hang attributes on it
        :param sigma_init: (float): Variance of the normal distribution from which the attribute vectors of each class
        are taken separately
        :param sigma_every: (float): Variance of noise added to the vector of attributes pf each class for every node
        :param m: Dimension of attributes

        :return: (nx.Graph): Attributed graph of networkx type
        """
        partition = community_louvain.best_partition(graph.graph, random_state=28)
        len_of_every_partition = {}
        for i in partition:
            if partition[i] not in len_of_every_partition:
                len_of_every_partition[partition[i]] = 1
            else:
                len_of_every_partition[partition[i]] += 1
        X = torch.normal(
            torch.zeros(len(len_of_every_partition), m),
            torch.ones(len(len_of_every_partition), m) * sigma_init,
        )

        for i in partition:
            attr = X[partition[i]] + torch.normal(
                torch.zeros(m), torch.ones(m) * sigma_every
            )
            graph.add_node(i, attribute=attr)
        return graph
