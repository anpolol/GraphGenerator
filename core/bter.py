import collections
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx import erdos_renyi_graph, expected_degree_graph
from scipy.stats import rv_discrete


class BTER:
    def __init__(
        self,
        degrees: List[int],
        etta: float = 0.1,
        ro: float = 0.7,
        d_manual: float = 0.75,
        betta: float = 0.1,
    ) -> None:
        """
        The BTER generator of graph

        :param degrees: ([int]): List of degrees of nodes in gnerated graphs
        :param etta: (float): The hyperparameter of BTER model (default: 0.1)
        :param ro: (float): The hyperparameter of BTER model (default: 0.7)
        :param d_manual: (float): The hyperparameter of BTER model (default: 0.75)
        :param betta: (float): The hyperparameter of BTER model (default: 0.1)
        """

        self.degrees = degrees
        self.etta = etta
        self.ro = ro
        self.d_manual = d_manual
        self.betta = betta
        super().__init__()

    def build_graph(self) -> nx.Graph:
        """
        Build graph of networkx.Graph type

        :return: (networkx.Graph): Generated graph of networkx.Graph type
        """
        graph = nx.Graph()

        filtered_nodes = []
        for deg in sorted(self.degrees):
            if deg > 0 and deg not in filtered_nodes:
                filtered_nodes.append(deg)

        graph.add_nodes_from(filtered_nodes)

        self.degrees = sorted(self.degrees)
        min_deg = self.degrees[0]

        degrees_except_min = []
        ones = 0
        for deg in self.degrees:
            if deg > min_deg:
                degrees_except_min.append(deg)
            elif deg == min_deg:
                ones += 1
        excesses = np.zeros(ones)
        excesses[int(np.round(ones * self.d_manual) + 1) :] = min_deg * 0.1
        if len(degrees_except_min) != 0:
            communities, mapping = self._making_communities(degrees_except_min, ones)
        excesses, graph = self._excesses(communities, mapping, excesses, graph)

        len_excesses = len(excesses)
        len_negative = 0
        len_less_min_deg = 0
        degs = []
        for deg in excesses:
            if deg <= 0:
                len_negative += 1
            if deg < min_deg + 1:
                len_less_min_deg += 1
            if deg > 0:
                degs.append(deg)

        if sum(degs) > 0:
            graph = self._random_edges(
                graph, degs, len_negative, len_excesses, len_less_min_deg, min_deg
            )
            # fixing excesses considering attached edges
            excesses = self._fix_excesses(excesses, ones, min_deg)

        # add remaining edges CL model
        if sum(excesses) > 0:
            graph = self._cl_model(graph, excesses, len_negative)
        return graph

    def _excesses(self, communities, mapping, excesses, graph):
        dmax = max(self.degrees)
        excesses = list(excesses)
        for i in communities:
            comm = communities[i]
            if i != max(communities):
                ro_r = self.ro * (
                    1 - self.etta * pow((np.log(comm[0] + 1) / np.log(dmax + 1)), 2)
                )

                g_er = erdos_renyi_graph(len(comm), ro_r)
                edges = []
                for e in g_er.edges():
                    edges.append((mapping[i][e[0]], mapping[i][e[1]]))

                graph.add_edges_from(edges)
                for deg in comm:
                    excesses.append(deg - ro_r * (len(comm) - 1))

            else:
                # last community with the highest degrees we do not fix
                for deg in comm:
                    excesses.append(deg)
        return excesses, graph

    def _fix_excesses(self, excesses, ones, min_deg):
        pmq = int(np.round(ones)) * min_deg
        rat = pmq / (pmq + sum(excesses))
        tetta = 1 - 2 * rat + self.betta

        excesses_new = []
        for ex in excesses:
            excesses_new.append(self.func_excesses(tetta, ex))
        return excesses_new

    def _cl_model(self, graph, excesses, len_negative):
        num_edges = int(np.round(sum(excesses) / 2))

        RandEdge = rv_discrete(
            len_negative,
            len(excesses) - 1,
            values=(
                range(len_negative, len(excesses)),
                self.pk_edge(excesses[len_negative:]),
            ),
        )
        for e in range(num_edges):
            edge_1 = RandEdge.rvs()
            edge_2 = RandEdge.rvs()
            graph.add_edge(edge_1, edge_2)
        return graph

    def _random_edges(
        self, graph, degs, len_negative, len_excesses, len_less_min_deg, min_deg
    ):
        RandNode = rv_discrete(
            len_negative,
            len_excesses - 1,
            values=(range(len_negative, len_excesses), self.pk_edge(degs)),
        )

        for i in range(len_less_min_deg):
            for h in range(int(min_deg)):
                graph.add_edge(i, RandNode.rvs())
        return graph

    def _making_communities(self, degrees_except_min, ones):
        communities = {}
        i = 0
        while len(degrees_except_min) > degrees_except_min[0]:
            c = int(degrees_except_min[0])
            communities[i] = degrees_except_min[: c + 1]

            i += 1
            degrees_except_min = degrees_except_min[c + 1:]
            if len(degrees_except_min) == 0:
                break
        if not len(degrees_except_min) == 0:
            communities[i] = degrees_except_min

        mapping = {}
        k = ones
        for i in communities:
            mapping[i] = {}
            for j, node in enumerate(communities[i]):
                mapping[i][j] = k + j
            k += len(communities[i])

        return communities, mapping

    def func_excesses(self, deg: float, tetta: float) -> float:
        """
        Improve values of excess degrees

        :param deg: (float): Value of degree
        :param tetta: (float): The correction ratio
        :return: (float): Improved value of degree
        """

        if deg > 0:  # >=2
            if tetta * deg >= 0:
                return tetta * deg
            else:
                return 0
        else:
            return deg

    def plot(self, graph: nx.Graph, degrees_old: List[int]) -> None:
        """
        Plot degree distributions of required degrees and of build_graphed Graph

        :param graph: (networkx.Graph): Built graph
        :param degrees_old: ([int]): List of required degrees
        """
        degrees_new = list(dict(graph.degree()).values())
        dic = dict()
        for deg in sorted(degrees_old):
            if deg not in dic:
                dic[deg] = 1
            else:
                dic[deg] += 1

        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(111)

        x = list(dic.keys())
        y = np.array(list(dic.values())).reshape(1, -1)
        ax1.scatter(x=x, y=y, label="Expected")

        dic = dict()
        for deg in sorted(degrees_new):
            if deg not in dic:
                dic[deg] = 1
            else:
                dic[deg] += 1

        x = list(dic.keys())
        y = np.array(list(dic.values())).reshape(1, -1)
        ax1.scatter(x=x, y=y, marker="+", color="green", label="Actual")
        ax1.legend(loc="upper center", shadow=True, fontsize="x-large")
        plt.show()

    def pk_edge(self, degrees: List[int]) -> Tuple[float]:
        """
        Build probability for distribution of edges in BTER

        :param degrees: ([int]): List of degrees
        :return: (Tuple(float)): Distribution
        """

        probs = []
        for deg in degrees:
            probability = deg / sum(degrees)
            probs.append(probability)
        return tuple(probs)
