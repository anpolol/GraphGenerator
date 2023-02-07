import collections
from collections import deque
from typing import Any, Dict, List, Tuple

import community as community_louvain
import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.stats import rv_discrete

from core.bter import BTER


class Main:
    def __init__(
        self,
        N: int,
        max_d: int,
        L: int,
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


        :param N: (int): Number of nodes in required graph
        :param max_d: (int): Degree value of the node with the maximum degree
        :param L: (int): Number of classes/labels in graph
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
        self.N = N
        self.max_d = max_d
        self.min_d = min_d
        self.L = L
        self.etta = etta
        self.ro = ro
        self.mu = mu
        self.power = power
        self.sigma_init = sigma_init
        self.sigma_every = sigma_every
        self.dim = dim
        self.CLASSES = sizes
        self.manual = manual
        self.d_manual = d_manual
        self.betta = betta
        super().__init__()

    # четыре функции ниже нужны для создания степенного распределения
    def xk(self, min_d: int, max_d: int) -> List[int]:
        """
        Build the list of uniform steps between min_d and max_d values

        :param min_d: (int): Degree value of the node with the minimum degree
        :param max_d: (int): Degree value of the node with the maximum degree
        :return: ([int]): list of uniform steps between min_d and max_d values
        """
        return range(min_d, max_d + 1)

    def su(self, min_d: int, max_d: int) -> float:
        """
        Calculate the sum of inverse power for power degree distribution

        :param min_d: (int): Degree value of the node with the minimum degree
        :param max_d: (int): Degree value of the node with the maximum degree
        :return: (float): The sum
        """
        su = 0
        for i in self.xk(min_d, max_d):
            su += 1 / (pow(i, self.power))
        return su

    def pk(self, min_d: int, max_d: int) -> Tuple[float]:
        """
        Build a power distribution of degrees

        :param min_d: (int): Degree value of the node with the minimum degree
        :param max_d: (int): Degree value of the node with the maximum degree
        :return: (Tuple[float]): The power degree distribution
        """
        l = []
        summ = self.su(min_d, max_d)
        for x in self.xk(min_d, max_d):
            ll = 1 / (pow(x, self.power) * summ)
            l.append(ll)
        return tuple(l)

    def making_degree_dist(
        self, min_d: int, max_d: int, N: int, mu: float
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Build three lists of degrees of nodes: overall, inside group and outside.

        :param min_d: (int): Degree value of the node with the minimum degree
        :param max_d: (int): Degree value of the node with the maximum degree
        :param N: (int): Number of nodes in the required graph
        :param mu: (float): required label assortativity
        :return: (([int],[int],[int])): Lists of degree of nodes, degrees of nodes inside their respective classes and degrees outside their own class
        """
        RandPL = rv_discrete(
            min_d, max_d, values=(self.xk(min_d, max_d), self.pk(min_d, max_d))
        )
        degrees = np.sort(RandPL.rvs(size=N))
        degrees_out = []
        degrees_in = (np.round(degrees * mu)).astype(np.int32)

        for j, deg in enumerate(degrees):
            degrees_out.append(deg - degrees_in[j])

        counter = collections.Counter(degrees)
        k = 0

        for i in range(1, int(np.ceil(1 / mu))):
            if i in counter:
                ones = len(list(filter(lambda x: x == i, degrees)))
                pr = torch.bernoulli(torch.ones(ones) * mu).numpy()
                degrees_in[k : k + ones] = pr * i
                degrees_out[k : k + ones] = (np.ones(ones) - pr) * i
                k = k + ones
        # else:
        #   ones = len(list(filter(lambda x:x==1, degrees)))
        #  if ones>0:
        #     pr = torch.bernoulli(torch.ones(ones)*mu).numpy()
        #    degrees_in[:ones] = pr
        #   degrees_out[:ones] = np.ones(ones) - pr

        return degrees, degrees_in, degrees_out

    def making_clusters(
        self, L: int, degrees_in: List[int]
    ) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
        """
        Build dictinary mappings of labels to degrees and nodes to labels

        :param L: (int): Number of classes
        :param degrees_in: ([int]): List of degrees inside respective group
        :return: (({int: int}, {int: int}, {int: int}))
        """
        # равномерный отбор
        labels_degrees = {}
        mapping = {}
        clusters = {}

        degrees_to_cluster = sorted(degrees_in)
        nodes = np.argsort(degrees_in)
        for j, (node, degree) in enumerate(list(zip(nodes, degrees_to_cluster))):
            if j % L not in labels_degrees:
                labels_degrees[j % L] = []

            labels_degrees[j % L].append(degree)
            clusters[node] = j % L
            ##todo
            if j % L not in mapping:
                mapping[j % L] = {}
                mapping[j % L][0] = node
            else:
                mapping[j % L][max(mapping[j % L].keys()) + 1] = node

        return labels_degrees, mapping, clusters  # clusters - лебл для кжадой вершины

    #!!! TODO Мб подумать как это возможно сделать покороче?
    def making_clusters_with_sizes(
        self, L: int, degrees_in: List[int], size_ratio: List[float]
    ) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:  # TODO
        """
        Make labels for nodes forcing the ratios of the sizes of classes according to size_ration list

        :param L: (int): Number of classes
        :param degrees_in: ([int]): List of degrees inside respective group
        :param size_ratio: ([float]): List of ratios of the sizes of classes of nodes
        :return: (({int: int}, {int: int}, {int: int}))
        """
        # равномерный отбор
        degrees_in = deque(degrees_in)

        labels_degrees = {}
        mapping = {}
        clusters = {}

        size_ratio[::-1].sort()
        sizes = np.round(np.array(size_ratio) * (len(degrees_in) / sum(size_ratio)))
        if sum(sizes) <= self.N - 1:
            sizes[0] += self.N - sum(sizes)

        for l in range(L):
            labels_degrees[l] = deque([])
            mapping[l] = {}

        list_of_classes = deque(range(L))  # содержит номера классов
        first_idx = (
            0  # первый СТАРЫЙ индекс. Надо для маппинга из нового индекса в старый
        )
        last_idx = len(degrees_in) - 1  # последний СТАРЫЙ индекс

        while len(list_of_classes) != 0:
            list_classes = list(list_of_classes)
            for l in list_classes:
                if len(labels_degrees[l]) < sizes[l] - 1:
                    mid = int(len(labels_degrees[l]) / 2)
                    labels_degrees[l].insert(mid, degrees_in.popleft())  # добавляю
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
            self.min_d, self.max_d, self.N, self.mu
        )

        if self.CLASSES is not None:
            labels_degrees, mapping, clusters = self.making_clusters_with_sizes(
                self.L, degrees_in, self.CLASSES
            )
        else:
            labels_degrees, mapping, clusters = self.making_clusters(self.L, degrees_in)

        self.G = nx.Graph()
        for j in range(self.N):
            self.G.add_node(j, label=clusters[j])

        # сначала собираем ребра с дргуими классами
        if self.manual == True:
            G_out = self.manual_out_degree(degrees_out, clusters)
            self.G.add_edges_from(G_out.edges())
        else:
            G_out, mapping_new2_to_new = self.bter_model_edges(
                degrees_out, self.etta, self.ro
            )
            for edge in G_out.edges():
                self.G.add_edge(
                    mapping_new2_to_new[edge[0]], mapping_new2_to_new[edge[1]]
                )
            # print(degrees_out, sorted(dict(G_out.degree()).values()))
        # теперь внутри классов собираем ребра
        for label in labels_degrees:
            degrees_in = labels_degrees[label]

            G_in, mapping_new2_to_new = self.bter_model_edges(
                degrees_in, self.etta, self.ro
            )

            for edge in G_in.edges():
                self.G.add_edge(
                    mapping[label][mapping_new2_to_new[edge[0]]],
                    mapping[label][mapping_new2_to_new[edge[1]]],
                )

        self.generate_attributes(self.dim)
        return self.G, clusters

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
            len(degrees_new),
            degrees_new,
            etta,
            ro,
            d_manual=self.d_manual,
            betta=self.betta,
        )
        G_model = model_degrees.construct()

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
        degrees_new = list(dict(self.G.degree()).values())
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
        legend = ax1.legend(loc="upper center", shadow=True, fontsize="x-large")
        plt.show()

    def statistics(self) -> Dict[Any, Any]:
        """
        Calculate characteritics of the graph

        :return: ({Any: Any}): Dictionary of calculated graph characteristics in view 'name_of_characteristic: value_of_this_characteristic'
        """
        dict_of_parameters = {
            "Power": self.power,
            "N": self.N,
            "M": self.max_d,
            "L": self.L,
            "Eta": self.etta,
            "Ro": self.ro,
            "Mu": self.mu,
            "Disper": self.sigma_init / self.sigma_every,
            "d": self.dim,
            "Avg Degree": np.mean(list(dict(self.G.degree()).values())),
            "Cluster": nx.average_clustering(self.G),
            "Density": nx.density(self.G),
            "Min degree": self.min_d,
        }

        feature_assort = 0
        label_assort = 0
        for i in self.G.nodes():
            s = 0
            s_l = 0
            t = 0
            for nei in self.G.neighbors(i):
                t += 1
                if (
                    self.cos(
                        self.G.nodes()[i]["attribute"], self.G.nodes()[nei]["attribute"]
                    )
                    > 0.5
                ):
                    s += 1
                if self.G.nodes()[nei]["label"] == self.G.nodes()[i]["label"]:
                    s_l += 1
            if t > 0:
                label_assort += s_l / t
                feature_assort += s / t

        dict_of_parameters["Feature Assort"] = feature_assort / len(self.G.nodes())
        dict_of_parameters["Label Assort"] = label_assort / len(self.G.nodes())
        dict_of_parameters["connected components"] = nx.number_connected_components(
            self.G
        )

        if nx.number_connected_components(self.G) == 1:
            iG = ig.Graph.from_networkx(self.G)
            avg_shortest_path = 0
            for i in iG.shortest_paths():
                for l in i:
                    avg_shortest_path += l
            avg_s_p = avg_shortest_path / (self.N * self.N - self.N)
        else:
            p = dict_of_parameters["connected components"]
            avg_shortes_path = 0
            for nodes in nx.connected_components(self.G):
                g = self.G.subgraph(nodes)
                g_ig = ig.Graph.from_networkx(g)
                n = g.number_of_nodes()

                avg = 0
                for i in g_ig.shortest_paths():
                    for l in i:
                        avg += l
                if n != 1:
                    avg_shortes_path += avg / (n * n - n)
                else:
                    avg_shortes_path = avg

            avg_s_p = avg_shortes_path / p
            # print(p)

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
            dict_of_parameters["N"],
            dict_of_parameters["M"],
            dict_of_parameters["Min degree"],
            dict_of_parameters["L"],
            dict_of_parameters["Eta"],
            dict_of_parameters["Ro"],
            dict_of_parameters["Mu"],
            dict_of_parameters["Disper"],
            dict_of_parameters["d"],
            dict_of_parameters["Avg Degree"],
            dict_of_parameters["Cluster"],
            dict_of_parameters["Density"],
            dict_of_parameters["Feature Assort"],
            dict_of_parameters["Label Assort"],
            dict_of_parameters["Avg shortest path"],
            dict_of_parameters["connected components"],
        ]
        row_series = pd.Series(to_append, index=df.columns)
        df = df.append(row_series, ignore_index=True)
        return df

    def print_statistics(self, dict_of_parameters: Dict[Any, Any]) -> None:
        """
        Print characteristics of the built Graph

        :param dict_of_parameters: ({Any: Any}): Parameters to add
        """
        print("PARAMETERTS ")
        print("--------------------")
        print("Power of power law", dict_of_parameters["Power"])
        print("Number of nodes: ", dict_of_parameters["N"])
        print("Max degree: ", dict_of_parameters["M"])
        print("Number of classes: ", dict_of_parameters["L"])
        print(
            "Etta:{}, ro:{}".format(dict_of_parameters["Eta"], dict_of_parameters["Ro"])
        )
        print("Ratio of neigbors with same label: ", dict_of_parameters["Mu"])
        print("Ratio of dispertions of attributes", dict_of_parameters["Disper"])
        print("Dimension of attributes: ", dict_of_parameters["d"])
        print("--------------------")
        print("PROPERTIES ")
        print("--------------------")
        print("Connected components: ", nx.number_connected_components(self.G))
        print("Average degree: ", dict_of_parameters["Avg Degree"])
        print("Cluster coef: ", dict_of_parameters["Cluster"])

        print("Feature assort: ", dict_of_parameters["Feature Assort"])
        print("Label assort:", dict_of_parameters["Label Assort"])
        print("Average shortest path", dict_of_parameters["Avg shortest path"])
        print("--------------------")

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
        # n_edges=int(np.round(sum(degrees_out)/2))
        # RandEdge_1 = rv_discrete(0,len(degrees_out)-1,values=(self.xke(0,len(degrees_out)),self.pk_edge(degrees_out)))

        # for e in range(n_edges):

        #   edge_1=RandEdge_1.rvs()
        #  edge_2=RandEdge_1.rvs()
        # if clusters[edge_1] == clusters[edge_2]:
        #    while clusters[edge_1] == clusters[edge_2]:
        #       edge_1=RandEdge_1.rvs()
        #      edge_2=RandEdge_1.rvs()
        # G_model.add_edge(edge_1,edge_2)

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
        partition = community_louvain.best_partition(self.G)
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
            self.G.add_node(i, attribute=attr)
