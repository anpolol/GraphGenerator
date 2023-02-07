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
        N: int,
        degs: List[int],
        etta: float = 0.1,
        ro: float = 0.7,
        d_manual: float = 0.75,
        betta: float = 0.1,
    ) -> None:
        """
        The BTER generator of graph

        :param N: (int): Number of nodes in constructed graph
        :param degs: ([int]): List of degrees of nodes in gnerated graphs
        :param etta: (float): The hyperparameter of BTER model (default: 0.1)
        :param ro: (float): The hyperparameter of BTER model (default: 0.7)
        :param d_manual: (float): The hyperparameter of BTER model (default: 0.75)
        :param betta: (float): The hyperparameter of BTER model (default: 0.1)
        """

        self.degrees = degs
        self.N = N
        self.etta = etta
        self.ro = ro
        self.d_manual = d_manual
        self.betta = betta
        super().__init__()

    def construct(self) -> nx.Graph:
        """
        Build graph of networkx.Graph type

        :return: (networkx.Graph): Generated graph of networkx.Graph type
        """
        G = nx.Graph()
        G.add_nodes_from(
            list(range(len(list(filter(lambda x: x > 0, sorted(self.degrees))))))
        )
        # со степенями 1 надо отдельно разбираться

        min_deg = self.degrees[0]
        degrees = list(filter(lambda x: x > min_deg, self.degrees))
        # с единициами работаем отдельно, excesses - оставшиеся степени
        ones = len(list(filter(lambda x: x == min_deg, self.degrees)))

        excesses = np.zeros(ones)
        excesses[int(np.round(ones * self.d_manual) + 1) :] = min_deg * 0.1

        if len(degrees) != 0:
            # разбиваем на общества
            communities = {}
            i = 0
            while len(degrees) > degrees[0]:
                c = int(degrees[0])
                communities[i] = degrees[: c + 1]

                i += 1
                degrees = degrees[c + 1 :]
                if len(degrees) == 0:
                    break
            if not len(degrees) == 0:
                communities[i] = degrees

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
                    g_ER = erdos_renyi_graph(len(comm), ro_r)
                    edges = []
                    for e in g_ER.edges():
                        edges.append((mapping[i][e[0]], mapping[i][e[1]]))

                    G.add_edges_from(edges)
                    for deg in comm:
                        excesses.append(deg - ro_r * (len(comm) - 1))
                    #   print('exc',deg,i,len(communities),len(communities[i]),deg - ro_r*(len(comm)-1))

                else:
                    # последнюю коммьнити с оч высокими степенями мы не трогаем
                    for deg in comm:
                        excesses.append(deg)
            # print('exc',excesses)
        # создаем ребра для 1
        l = len(list(filter(lambda x: x <= 0, excesses)))  # ==0 <2
        ll = len(list(filter(lambda x: x < min_deg + 1, excesses)))
        r = len(excesses)
        degs = list(filter(lambda x: x > 0, excesses))  # >0 >=2
        # print(sum(self.pk_edge(degs)),(degs))
        if sum(degs) > 0:
            # print(excesses,degs)
            RandEdge = rv_discrete(
                l, r - 1, values=(self.xke(l, r), self.pk_edge(degs))
            )

            for i in range(ll):
                for h in range(int(min_deg)):
                    G.add_edge(i, RandEdge.rvs())

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
            n_edges = int(np.round(sum(excesses) / 2))

            RandEdge_2 = rv_discrete(
                l,
                len(excesses) - 1,
                values=(self.xke(l, len(excesses)), self.pk_edge(excesses[l:])),
            )
            for e in range(n_edges):
                edge_1 = RandEdge_2.rvs()
                edge_2 = RandEdge_2.rvs()
                G.add_edge(edge_1, edge_2)
        return G

    def func_excesses(self, x: float, tetta: float) -> float:
        """
        Improve values of excess degrees

        :param x: (float): Value of degree
        :param tetta: (float): The correction ratio
        :return: (float): Improved value of degree
        """

        if x > 0:  # >=2
            if tetta * x >= 0:
                return tetta * x
            else:
                return 0
        else:
            return x

    def plot(self, G: nx.Graph, degrees_old: List[int]) -> None:
        """
        Plot degree distributions of required degrees and of constructed Graph

        :param G: (networkx.Graph): Built graph
        :param degrees_old: ([int]): List of required degrees
        """
        degrees_new = list(dict(G.degree()).values())
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
        legend = ax1.legend(loc="upper center", shadow=True, fontsize="x-large")
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
        l = []
        for d in degrees:
            ll = d / sum(degrees)
            l.append(ll)
        return tuple(l)
