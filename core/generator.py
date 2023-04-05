import collections
from collections import deque
from typing import Any, Dict, List, Tuple

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import torch
from community import community_louvain
from matplotlib import pyplot as plt
from scipy.stats import rv_discrete

from core.bter import BTER


class Main:
    def __init__(
        self,
        num_nodes: int,
        max_d: int,
        num_classes: int,
        etta: float,
        ro: float,
        mu: float,
        sigma_init: float,
        sigma_every: float,
        dim: float,
        power: float = 2.0,
        sizes: List[int] = None,
        manual: bool = False,
        min_d: int = 1,
        d_manual: float = 0.75,
        betta: float = 0.1,
    ) -> None:
        """
        The generator for graphs with controllable graph characteristics based on BTER model


        :param num_nodes: (int): Number of nodes in required graph
        :param max_d: (int): Degree value of the node with the maximum degree
        :param num_classes: (int): Number of classes/labels in graph
        :param etta: (float): The hyperparameter for BTER
        :param ro: (float): The hyperparameter for BTER
        :param mu: (float): Required label assortativity
        :param sigma_init: (float): Variance of the normal distribution from which the attribute vectors of each class are taken separately
        :param sigma_every: (float): Variance of noise added to the vector of attributes pf each class for every node
        :param dim: (float): Dimension of attribute vector
        :param power: (float): The power of the degree distribution (default: 2)
        :param sizes: ([int]): Sizes of classes, if None, the size will be approximately the same (default: None)
        :param manual: (bool): Flag identifying the way of connecting edges between classes -- mannually or using BTER
        :param min_d: (int): Degree value of the node with the minimum degree
        :param d_manual: (float): The hyperparameter of BTER model (default: 0.75)
        :param betta: (float): The hyperparameter of BTER model (default: 0.1)
        """
        self.num_nodes = num_nodes
        self.max_d = max_d
        self.min_d = min_d
        self.num_classes = num_classes
        self.etta = etta
        self.ro = ro
        self.mu = mu
        self.power = power
        self.sigma_init = sigma_init
        self.sigma_every = sigma_every
        self.dim = dim
        self.class_distr = sizes
        self.manual = manual
        self.d_manual = d_manual
        self.betta = betta
        super().__init__()

    def sum(self, min_d: int, max_d: int) -> float:
        """
        Calculate the sum of inverse power for power degree distribution

        :param min_d: (int): Degree value of the node with the minimum degree
        :param max_d: (int): Degree value of the node with the maximum degree
        :return: (float): The sum
        """
        sum = 0
        for i in range(min_d, max_d + 1):
            sum += 1 / (pow(i, self.power))
        return sum

    def pk(self, min_d: int, max_d: int) -> Tuple[float]:
        """
        Build a power distribution of degrees

        :param min_d: (int): Degree value of the node with the minimum degree
        :param max_d: (int): Degree value of the node with the maximum degree
        :return: (Tuple[float]): The power degree distribution
        """
        probs = []
        sum = self.sum(min_d, max_d)
        for x in range(min_d, max_d + 1):
            probability = 1 / (pow(x, self.power) * sum)
            probs.append(probability)
        return tuple(probs)

    def making_degree_dist(
        self, min_d: int, max_d: int, num_nodes: int, mu: float
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Build three lists of degrees of nodes: overall, inside group and outside.

        :param min_d: (int): Degree value of the node with the minimum degree
        :param max_d: (int): Degree value of the node with the maximum degree
        :param num_nodes: (int): Number of nodes in the required graph
        :param mu: (float): required label assortativity
        :return: (([int],[int],[int])): Lists of degree of nodes, degrees of nodes inside their respective classes and degrees outside their own class
        """
        rand_power_law = rv_discrete(
            min_d, max_d, values=(range(min_d, max_d + 1), self.pk(min_d, max_d))
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
                degrees_in[k : k + ones] = prob * i
                degrees_out[k : k + ones] = (np.ones(ones) - prob) * i
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
        self, num_classes: int, degrees_in: List[int], size_ratio: List[float]
    ) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
        """
        Make labels for nodes forcing the ratios of the sizes of classes according to size_ration list

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
        if sum(sizes) <= self.num_nodes - 1:
            sizes[0] += self.num_nodes - sum(sizes)

        for l in range(num_classes):
            labels_degrees[l] = deque([])
            mapping[l] = {}

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

    def making_graph(self) -> Tuple[nx.Graph, Dict[int, int]]:
        """
        Generate graph, main function

        :return: ((networkx.Graph, {int: int})): Graph if type networkx.Graph and mapping nodes to labels
        """
        self.degrees, degrees_in, degrees_out = self.making_degree_dist(
            self.min_d, self.max_d, self.num_nodes, self.mu
        )

        if self.class_distr is not None:
            labels_degrees, mapping, clusters = self.making_clusters_with_sizes(
                self.num_classes, degrees_in, self.class_distr
            )
        else:
            labels_degrees, mapping, clusters = self.making_clusters(
                self.num_classes, degrees_in
            )

        self.graph = nx.Graph()
        for j in range(self.num_nodes):
            self.graph.add_node(j, label=clusters[j])

        # first collect edges with other classes
        if self.manual == True:
            G_out = self.manual_out_degree(degrees_out, clusters)
            self.graph.add_edges_from(G_out.edges())
        else:
            G_out, mapping_new2_to_new = self.bter_model_edges(
                degrees_out, self.etta, self.ro
            )
            for edge in G_out.edges():
                self.graph.add_edge(
                    mapping_new2_to_new[edge[0]], mapping_new2_to_new[edge[1]]
                )

        # now inside the classes we collect edges
        for label in labels_degrees:
            degrees_in = labels_degrees[label]

            G_in, mapping_new2_to_new = self.bter_model_edges(
                degrees_in, self.etta, self.ro
            )

            for edge in G_in.edges():
                self.graph.add_edge(
                    mapping[label][mapping_new2_to_new[edge[0]]],
                    mapping[label][mapping_new2_to_new[edge[1]]],
                )

        self.generate_attributes(self.dim)
        return self.graph, clusters

    def bter_model_edges(
        self, degrees: List[int], etta: float, ro: float
    ) -> Tuple[nx.Graph, Dict[int, int]]:
        """
        Add edges to nodes with degrees

        :param degrees: ([int]): Degrees of nodes
        :param etta: (float): The hyperparameter for BTER model
        :param ro: (float): The hyperparameter for BTER model
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
        model_degrees = BTER(
            degrees_new,
            etta,
            ro,
            d_manual=self.d_manual,
            betta=self.betta,
        )
        G_model = model_degrees.build_graph()

        return G_model, mapping_new2_to_new

    def cos(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Calculate cos between two vectors a and b

        :param a: (torch.Tensor): First tensor
        :param b: (torch.Tensor): Second tensor
        :return: torch.Tensor: One value of cos between a and b
        """
        return (torch.matmul(a, b)) / (torch.norm(a) * torch.norm(b))

    def plot_dist(self) -> None:
        """
        Plot expected degree disttribution and real
        """
        degrees_new = list(dict(self.graph.degree()).values())
        dic = dict()
        for deg in sorted(self.degrees):
            if deg not in dic:
                dic[deg] = 1
            else:
                dic[deg] += 1

        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(111)

        x = list(dic.keys())
        y = np.array(list(dic.values())).reshape(1, -1)
        ax1.scatter(x=x, y=y, label="Expected Degree Distribution")

        dic = dict()
        for deg in sorted(degrees_new):
            if deg not in dic:
                dic[deg] = 1
            else:
                dic[deg] += 1

        x = list(dic.keys())
        y = np.array(list(dic.values())).reshape(1, -1)
        ax1.scatter(
            x=x, y=y, marker="+", color="green", label="Actual Degree Distribution"
        )
        ax1.legend(loc="upper center", shadow=True, fontsize="x-large")
        plt.show()

    def statistics(self) -> Dict[Any, Any]:
        """
        Calculate characteritics of the graph

        :return: ({Any: Any}): Dictionary of calculated graph characteristics in view 'name_of_characteristic:
        value_of_this_characteristic'
        """
        dict_of_parameters = {
            "Power": self.power,
            "Number of nodes": self.num_nodes,
            "Max degree": self.max_d,
            "Number of classes": self.num_classes,
            "Eta": self.etta,
            "Ro": self.ro,
            "Mu": self.mu,
            "Disper": self.sigma_init / self.sigma_every,
            "Dimension": self.dim,
            "Avg Degree": np.mean(list(dict(self.graph.degree()).values())),
            "Cluster": nx.average_clustering(self.graph),
            "Density": nx.density(self.graph),
            "Min degree": self.min_d,
        }

        feature_assort = 0
        label_assort = 0
        for i in self.graph.nodes():
            s = 0
            s_l = 0
            t = 0
            for neigbour in self.graph.neighbors(i):
                t += 1
                if (
                    self.cos(
                        self.graph.nodes()[i]["attribute"],
                        self.graph.nodes()[neigbour]["attribute"],
                    )
                    > 0.5
                ):
                    s += 1
                if (
                    self.graph.nodes()[neigbour]["label"]
                    == self.graph.nodes()[i]["label"]
                ):
                    s_l += 1
            if t > 0:
                label_assort += s_l / t
                feature_assort += s / t

        dict_of_parameters["Feature Assort"] = feature_assort / len(self.graph.nodes())
        dict_of_parameters["Label Assort"] = label_assort / len(self.graph.nodes())
        dict_of_parameters["Connected components"] = nx.number_connected_components(
            self.graph
        )

        if nx.number_connected_components(self.graph) == 1:
            iG = ig.Graph.from_networkx(self.graph)
            avg_shortest_path = 0
            for shortest_path in iG.shortest_paths():
                for sp in shortest_path:
                    avg_shortest_path += sp
            avg_s_p = avg_shortest_path / (
                self.num_nodes * self.num_nodes - self.num_nodes
            )
        else:
            connected_components = dict_of_parameters["Connected components"]
            avg_shortes_path = 0
            for nodes in nx.connected_components(self.graph):
                g = self.graph.subgraph(nodes)
                g_ig = ig.Graph.from_networkx(g)
                num_nodes = g.number_of_nodes()

                avg = 0
                for shortes_paths in g_ig.shortest_paths():
                    for sp in shortes_paths:
                        avg += sp
                if num_nodes != 1:
                    avg_shortes_path += avg / (num_nodes * num_nodes - num_nodes)
                else:
                    avg_shortes_path = avg

            avg_s_p = avg_shortes_path / connected_components

        dict_of_parameters["Avg shortest path"] = avg_s_p

        return dict_of_parameters

    def pandas_stat(
        self, df: pd.DataFrame, dict_of_parameters: Dict[Any, Any]
    ) -> pd.DataFrame:
        """
        Add statistics of generated graph characteristics to the pandas DataFrame

        :param df: (pd.DataFrame): Initial Data Frame to which information should be added
        :param dict_of_parameters: ({Any: Any}): Parameters to add
        :return: (pd.DataFrame): Data frame with added information
        """
        to_append = [
            dict_of_parameters["Power"],
            dict_of_parameters["Number of nodes"],
            dict_of_parameters["Max degree"],
            dict_of_parameters["Min degree"],
            dict_of_parameters["Number of classes"],
            dict_of_parameters["Eta"],
            dict_of_parameters["Ro"],
            dict_of_parameters["Mu"],
            dict_of_parameters["Disper"],
            dict_of_parameters["Dimension"],
            dict_of_parameters["Avg Degree"],
            dict_of_parameters["Cluster"],
            dict_of_parameters["Density"],
            dict_of_parameters["Feature Assort"],
            dict_of_parameters["Label Assort"],
            dict_of_parameters["Avg shortest path"],
            dict_of_parameters["Connected components"],
        ]
        row_series = pd.Series(to_append, index=df.columns)
        df = df.append(row_series, ignore_index=True)
        return df

    def print_statistics(self, dict_of_parameters: Dict[Any, Any]) -> None:
        """
        Print characteristics of the built Graph

        :param dict_of_parameters: ({Any: Any}): Parameters to add
        """

        print(
            "PARAMETERTS: \n-------------------- \nPower of power law: {} \nNumber of nodes: {} \nMax degree: {} "
            "\nNumber of classes: {} \nEtta: {} \nro: {} \nRatio of neigbors with same label: {} "
            "\nRatio of variances of attributes: {} \nDimension of attributes: {} \n-------------------- "
            "\n".format(
                dict_of_parameters["Power"],
                dict_of_parameters["Number of nodes"],
                dict_of_parameters["Max degree"],
                dict_of_parameters["Number of classes"],
                dict_of_parameters["Eta"],
                dict_of_parameters["Ro"],
                dict_of_parameters["Mu"],
                dict_of_parameters["Disper"],
                dict_of_parameters["Dimension"],
            )
        )

        print(
            "PROPERTIES: \n-------------------- \nConnected components:{} \nAverage degree:{} \nCluster coef:{} "
            "\nFeature assort:{} \nLabel assort:{} \nAverage shortest path:{} "
            "\n--------------------".format(
                nx.number_connected_components(self.graph),
                dict_of_parameters["Avg Degree"],
                dict_of_parameters["Cluster"],
                dict_of_parameters["Feature Assort"],
                dict_of_parameters["Label Assort"],
                dict_of_parameters["Avg shortest path"],
            )
        )

    def draw_graph(self):
        nx.draw(self.graph)

    def manual_out_degree(
        self, degrees_out: List[int], clusters: Dict[int, int]
    ) -> nx.Graph:
        """
        Calculate edges between differenet classes in manual regime

        :param degrees_out: ([int]): List of degrees of nodes to different classes
        :param clusters: ({int: int}): Mapping of nodes into labels
        :return: (networkx.Graph): Constructed graph on out degrees of type networkx.Graph
        """
        G_model = nx.Graph()

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

    def generate_attributes(self, m: int) -> None:
        """
        Add attributes to nodes in the Graph

        :param m: Dimension of attributes
        """
        partition = community_louvain.best_partition(self.graph)
        len_of_every_partition = {}
        for i in partition:
            if partition[i] not in len_of_every_partition:
                len_of_every_partition[partition[i]] = 1
            else:
                len_of_every_partition[partition[i]] += 1
        X = torch.normal(
            torch.zeros(len(len_of_every_partition), m),
            torch.ones(len(len_of_every_partition), m) * self.sigma_init,
        )

        for i in partition:
            attr = X[partition[i]] + torch.normal(
                torch.zeros(m), torch.ones(m) * self.sigma_every
            )
            self.graph.add_node(i, attribute=attr)
