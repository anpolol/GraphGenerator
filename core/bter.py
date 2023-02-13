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
        graph.add_nodes_from(
            list(range(len(list(filter(lambda x: x > 0, sorted(self.degrees))))))
        )
        # со степенями 1 надо отдельно разбираться

        self.degrees = sorted(self.degrees)
        min_deg = self.degrees[0]

        degrees_except_min = list(filter(lambda x: x > min_deg, self.degrees))
        # с единициами работаем отдельно, excesses - оставшиеся степени
        ones = len(list(filter(lambda x: x == min_deg, self.degrees)))

        excesses = np.zeros(ones)
        excesses[int(np.round(ones * self.d_manual) + 1) :] = min_deg * 0.1

        if len(degrees_except_min) != 0:
            # разбиваем на общества
            communities = {}
            i = 0
            while len(degrees_except_min) > degrees_except_min[0]:
                c = int(degrees_except_min[0])
                communities[i] = degrees_except_min[: c + 1]

                i += 1
                degrees_except_min = degrees_except_min[c + 1 :]
                if len(degrees_except_min) == 0:
                    break
            if not len(degrees_except_min) == 0:
                communities[i] = degrees_except_min

            # маппинги
            mapping = {}
            k = ones
            for i in communities:
                mapping[i] = {}
                for j, node in enumerate(communities[i]):
                    mapping[i][j] = k + j
                k += len(communities[i])

            # внутри каждого коммьюнити мы делаем ребра по Erdos-Ernyi моедли
            # а excesses это "оставшиеся степени" вершин

            dmax = max(self.degrees)
            excesses = list(excesses)
            for i in communities:
                comm = communities[i]
                if i != max(communities):
                    ro_r = self.ro * (
                        1 - self.etta * pow((np.log(comm[0] + 1) / np.log(dmax + 1)), 2)
                    )
                    # ro_r=self.ro
                    # print('ro_r',ro_r)
                    g_er = erdos_renyi_graph(len(comm), ro_r)
                    edges = []
                    for e in g_er.edges():
                        edges.append((mapping[i][e[0]], mapping[i][e[1]]))

                    graph.add_edges_from(edges)
                    for deg in comm:
                        excesses.append(deg - ro_r * (len(comm) - 1))
                    #   print('exc',deg,i,len(communities),len(communities[i]),deg - ro_r*(len(comm)-1))

                else:
                    # последнюю коммьнити с оч высокими степенями мы не трогаем
                    for deg in comm:
                        excesses.append(deg)
            # print('exc',excesses)
        # создаем ребра для 1
        len_negative = len(list(filter(lambda x: x <= 0, excesses)))  # ==0 <2
        len_less_min_deg = len(list(filter(lambda x: x < min_deg + 1, excesses)))
        len_excesses = len(excesses)
        degs = list(filter(lambda x: x > 0, excesses))  # >0 >=2
        # print(sum(self.pk_edge(degs)),(degs))
        if sum(degs) > 0:
            # print(excesses,degs)
            RandNode = rv_discrete(
                len_negative, len_excesses - 1, values=(self.xke(len_negative, len_excesses), self.pk_edge(degs))
            )

            for i in range(len_less_min_deg):
                for h in range(int(min_deg)):
                    graph.add_edge(i, RandNode.rvs())

            # исправляем excesses с учетом присоединенных ребер
            tetta = (
                1
                - 2
                * (
                    int(np.round(ones))
                    * min_deg
                    / (int(np.round(ones)) * min_deg + sum(excesses))
                )
                + self.betta
            )
            excesses = list(map(lambda x: self.func_excesses(tetta, x), excesses))
            # print(excesses)
        if sum(excesses) > 0:
            # добавляем оставшиеся ребра CL моделью
            num_edges = int(np.round(sum(excesses) / 2))

            RandEdge = rv_discrete(
                len_negative,
                len(excesses) - 1,
                values=(self.xke(len_negative, len(excesses)), self.pk_edge(excesses[len_negative:])),
            )
            for e in range(num_edges):
                edge_1 = RandEdge.rvs()
                edge_2 = RandEdge.rvs()
                graph.add_edge(edge_1, edge_2)
        return graph

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
        # ax.plot(x_s,y_s)
        ax1.legend(loc="upper center", shadow=True, fontsize="x-large")
        plt.show()

    def xke(self, l: int, m: int) -> List[int]:
        """
        Build range of integers btween two values

        :param l: (int): First int
        :param m: (int): Second int
        :return: ([int]): List of integers between l and m excluding m
        """
        return range(l, m)

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
