import networkx as nx
import torch
from matplotlib import pyplot as plt
import numpy as np
from typing import Dict, Any
import pandas as pd
import igraph as ig


class Graph(nx.Graph):
    def __init__(self):
        super(Graph, self).__init__()

    def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Calculate cos between two vectors a and b

        :param a: (torch.Tensor): First tensor
        :param b: (torch.Tensor): Second tensor
        :return: torch.Tensor: One value of cos between a and b
        """
        return (torch.matmul(a, b)) / (torch.norm(a) * torch.norm(b))

    def save(self):
        pass

    def to_torch(self):
        pass

    def plot_dist(self, graph, expected_degrees) -> None:
        """
        Plot expected degree disttribution and real
        """
        degrees_new = list(dict(graph.degree()).values())
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


    def get_statistics(self, output_method: str='dict') -> Dict[Any, Any]: #TODO
        """
        Calculate characteritics of the graph

        :param: (str): The way of output of statistics, either 'dict' or 'pandas'. (default: dict)
        :return: ({Any: Any}): Dictionary of calculated graph characteristics in view 'name_of_characteristic:
        value_of_this_characteristic'
        """
        dict_of_parameters = {
            "Number of nodes": self.number_of_nodes(),
            "Max degree": max(self.degree()),
            "Number of classes": self.num_classes,
            "Dimension": self.dim,
            "Avg Degree": np.mean(list(dict(self.graph.degree()).values())),
            "Cluster": nx.average_clustering(self.graph),
            "Density": nx.density(self.graph),
            "Min degree": self.min_d,
        }

        feature_assort = 0
        label_assort = 0
        for i in self.nodes():
            s = 0
            s_l = 0
            t = 0
            for neigbour in self.neighbors(i):
                t += 1
                if (
                        self.cos(
                            self.nodes()[i]["attribute"],
                            self.nodes()[neigbour]["attribute"],
                        )
                        > 0.5
                ):
                    s += 1
                if (
                        self.nodes()[neigbour]["label"]
                        == self.nodes()[i]["label"]
                ):
                    s_l += 1
            if t > 0:
                label_assort += s_l / t
                feature_assort += s / t

        dict_of_parameters["Feature Assort"] = feature_assort / len(self.nodes())
        dict_of_parameters["Label Assort"] = label_assort / len(self.nodes())
        dict_of_parameters["Connected components"] = nx.number_connected_components(
            self.graph
        ) # TODO

        if nx.number_connected_components(self.graph) == 1:
            iG = ig.Graph.from_networkx(self.graph)
            avg_shortest_path = 0
            for shortest_path in iG.shortest_paths():
                for sp in shortest_path:
                    avg_shortest_path += sp
            avg_s_p = avg_shortest_path / (
                    self.number_of_nodes() * self.number_of_nodes() - self.number_of_nodes()
            )
        else:
            connected_components = dict_of_parameters["Connected components"]
            avg_shortes_path = 0
            for nodes in nx.connected_components(self.graph):
                g = self.subgraph(nodes)
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
