# from torch_geometric.data import Data
import collections
import os
import pickle
from typing import Any, AnyStr, Dict, Union

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from core.utils import cos


class Graph:
    def __init__(self):
        self.graph = nx.Graph()
        self.dict_of_parameters = dict()
        super(Graph, self).__init__()

    def edges(self):
        return self.graph.edges()

    def add_node(self, node, **attr):
        self.graph.add_node(node, **attr)

    def add_edge(self, v1, v2):
        self.graph.add_edge(v1, v2)

    def add_nodes_from(self, node_list):
        self.graph.add_nodes_from(node_list)

    def add_edges_from(self, edges_list):
        self.graph.add_edges_from(edges_list)

    def attr_labels_edges_count(self):
        try:
            self.graph.number_of_nodes() == len(self.labels)
        except:
            self.node_attr = []
            for i, attr in self.graph.nodes("attribute"):
                self.node_attr.append(attr.tolist())
            self.labels = []
            for i, lab in self.graph.nodes("label"):
                self.labels.append(lab)
            self.edges = self.graph.edges()

    def save(self, path, name: str):
        self.attr_labels_edges_count()
        labels = np.array(self.labels)
        np.save(
            str(path) + str(name) + "_edgelist.npy",
            np.array(self.edges),
        )
        np.save(
            str(path) + str(name) + "_attr.npy",
            self.node_attr,
        )
        np.save("../dataset/graph_" + str(name) + "_labels.npy", labels)
        with open("../dataset/graph_" + str(name) + ".pickle", "wb") as f:
            pickle.dump(self.graph, f)

    def to_torch(self):
        self.attr_labels_edges_count()
        print(
            "x",
            torch.tensor(self.node_attr),
            "y",
            torch.tensor(self.labels),
            "edge_index",
            torch.tensor(self.edges),
        )
        data = Data(
            x=torch.tensor(self.node_attr),
            y=torch.tensor(self.labels),
            edge_index=torch.tensor(self.edges),
        )
        return data

    def plot_dist(self, expected_degrees) -> None:
        """
        Plot expected degree distribution and actual degree
        """
        degrees_new = list(dict(self.graph.degree()).values())
        dic = dict()
        for deg in sorted(expected_degrees):
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

    def get_statistics(
        self, generator_params: Dict[AnyStr, Any], output_method: str = "dict"
    ) -> Dict[Any, Any]:  # TODO
        """
        Calculate characteritics of the graph

        :param: (str): The way of output of statistics, either 'dict' or 'pandas'. (default: dict)
        :return: ({Any: Any}): Dictionary of calculated graph characteristics in view 'name_of_characteristic:
        value_of_this_characteristic'
        """
        if len(self.dict_of_parameters) == 0:
            self.attr_labels_edges_count()

            dim = generator_params["dim"]
            min_d = generator_params["min_d"]
            power = generator_params["power"]
            eta = generator_params["eta"]
            rho = generator_params["rho"]
            mu = generator_params["mu"]
            disper = generator_params["sigma_init"]

            num_classes = len(collections.Counter(self.labels))

            dict_of_parameters = {
                "Number of nodes": self.graph.number_of_nodes(),
                "Max degree": max(self.graph.degree()),
                "Power": power,
                "Eta": eta,
                "Rho": rho,
                "Mu": mu,
                "Disper": disper,
                "Number of classes": num_classes,
                "Dimension": dim,
                "Avg Degree": np.mean(list(dict(self.graph.degree()).values())),
                "Cluster": nx.average_clustering(self.graph),
                "Density": nx.density(self.graph),
                "Min degree": min_d,
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
                        cos(
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

            dict_of_parameters["Feature Assort"] = feature_assort / len(
                self.graph.nodes()
            )
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
                    self.graph.number_of_nodes() * self.graph.number_of_nodes()
                    - self.graph.number_of_nodes()
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
            self.dict_of_parameters = dict_of_parameters

        if output_method == "dict":
            return self.dict_of_parameters
        elif output_method == "pandas":
            return pd.from_dict(self.dict_of_parameters)

    def print_statistics(self, generator_params) -> None:
        """
        Print characteristics of the built Graph

        :param dict_of_parameters: ({Any: Any}): Parameters to add
        """
        dict_of_parameters = self.get_statistics(
            generator_params=generator_params, output_method="dict"
        )

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
                dict_of_parameters["Rho"],
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
