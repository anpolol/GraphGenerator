import copy
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, atpe, fmin, hp, tpe
from numpy.typing import ArrayLike
import sys
from core.generator import Main as Model
from core.agm_generator import AGM, PreferentialAttachment, SBM
import random

class TuneParameters:
    def __init__(self, number_of_trials: int, characteristics_check: List) -> None:
        """
        Class for tuning input parameters of Generator so that graph characteristics are equal to required

        :param number_of_trials (int): Number of minimum number of trials for tuning each required graph
        :param characteristics_check (List): List of lists of required graphs characteristics, each row represents characteristics
        for one graph, columns are follows: label assortativity, feature assortativity, clustering coefficient, average
        length of shortest paths, average degree
        """
        super(TuneParameters, self).__init__()

        # Setting up hyperparameters space
        self.hp_space = {
            "N": hp.quniform("N", 100, 10000, 100),
            "L": hp.quniform("L", 2, 20, 1),
            "inside_prob": hp.uniform("inside_prob", 0.1, 0.9),
            "outside_prob": hp.uniform("outside_prob", 0.1, 0.9),
            "number_of_groups": hp.quniform("number_of_groups", 1, 10, 1),
            "k0": hp.quniform("k0", 2, 10, 1),
            "k": hp.quniform("k", 2, 10, 1),
            "assort_corr_in": hp.uniform("assort_corr_in", 0.1, 0.9),
            "assort_corr_between": hp.uniform("assort_corr_between", 0.1, 0.9),
        }


        self.OUT_PAR_NAMES = [
            "assortativity",
            "Cluster",
            "Avg shortest path",
            "Avg Degree",
            "Connected components",
        ]
        self.number_of_trials = number_of_trials
        self.characteristics = characteristics_check
        self.characteristics_check = copy.deepcopy(characteristics_check)
        self.num_par_out = len(self.OUT_PAR_NAMES)
        self.trials = Trials()
        self.max_eval = 0
        self.limits = {
            "assortativity": 0.1,
            "Cluster": 0.05,
            "Avg shortest path": 0.3,
            "Avg Degree": 1.5,
            "Connected components": 10,
        }
        self.df_bench = pd.DataFrame(
            columns=[
                "assortativity",
                "Cluster",
                "Avg shortest path",
                "Avg Degree",
                "Connected components",
            ]
        )

    def chars_to_array(self, chars: Dict[str, float]) -> ArrayLike:
        """
        Convert dict to array for certain characteristics:  "Label Assort", "Feature Assort", "Cluster",
        "Avg shortest path", "Avg Degree", "Connected components"

        :param chars (Dict[str, float]): Dict of required graph characteristics
        :return: (Array): Array of charactristics
        """
        rez = []
        for p in self.OUT_PAR_NAMES:
            if p not in chars.keys():
                return None
            rez.append(chars[p])
        rez = np.array(rez)
        return rez

    def array_to_chars(self, arr: ArrayLike) -> Dict[str, float]:
        """
        Convert array to dict of required characteristics

        :param: arr (Array): Array of characteristics to convert
        :return: (Dict[str, float]): Dict of required characteristics
        """
        return {self.OUT_PAR_NAMES[i]: arr[i] for i in range(len(self.OUT_PAR_NAMES))}

    def loss_func(self, pred: List, target: List) -> float:
        """
        Count loss function: absolute error between target characteristics and built ones

        :param: pred (List): Characteristics of built graph
        :param: target (List): Required graph characteristics
        :return: (float): Loss function value
        """

        weight = np.ones(self.num_par_out)
        weight[-1] = 0.3

        print("loss func", (abs(((pred - target) / target)) * weight).sum())

        return (abs(((pred - target) / target)) * weight).sum()

    def early_stop(self, trials: Dict[Any, Any]) -> (bool, List):
        """
        Stop tuning if characteristics are reached

        :param trials (Dict[Any, Any]): Trials of hyperopt
        :return: (bool, List): True if stop is required, else False and empty list
        """
        out_pars = self.chars_to_array(trials.trials[-1]["result"])

        for num, targets_check in enumerate(self.characteristics):
            diff = np.abs((out_pars - targets_check))
            name = "".join(list(map(lambda x: str(x), self.targets[:-1])))
            if np.all(diff < self.chars_to_array(self.limits)) or os.path.exists(
                "../dataset/graph_" + str(name) + ".pickle"
            ):
                return True, []
        return False, []

    def objective(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Objective to minimize while searching parameters of gnerator which will build graph with required
        characteristics

        :param args (Dict[str, Any]): Dict of input parameters of generator which should be considered in current trial
        :return (Dict[str, Any]): Dict of required graph characteristics and loss value for current trial
        """

        probs = np.diag(int(args["k"]) * [args["inside_prob"]])
        probs = np.where(probs == 0, args["inside_prob"], args["outside_prob"])
        probs = probs.tolist()

        sample_labels_keys = list(range(int(args["N"])))
        sample_labels_items = random.choices(
            list(range(int(args["L"]))), k=int(args["N"])
        )

        sample_labels = dict(zip(sample_labels_keys, sample_labels_items))

        # Now for the AGM steps.  First, just create the AR method using the given data, the proposing distribution,
        # and the random sample of neighbors.

        # Now we actually do AGM!  Just plug in your proposing distribution (FCL Example Given) as well as
        # the edge_acceptor, and let it go!

        pa = PreferentialAttachment(int(args["N"]), int(args["k0"]), int(args["k"]))
        vertices, degree_dist = pa.Sample()
        # generator = FastChungLu(vertices, degree_dist)
        generator = SBM(
            vertices, degree_dist, int(args["number_of_groups"]), probs, int(args["N"])
        )

        agm = AGM(
            degree_dist,
            args["assort_corr_in"],
            args["assort_corr_between"],
            int(args["L"]),
        )
        agm_sample = agm.sample_graph(generator, sample_labels)

        # This should be fairly close to the initial graph's correlation
        # print('AGM Graph Correlation', ComputeLabelCorrelation(agm_sample, sample_labels))
        # o_append = pd.Series(
        #   [dict_statistics['Avg Degree'], dict_statistics['Cluster'], dict_statistics['Connected components'],
        #     dict_statistics['Avg shortest path'], ComputeLabelCorrelation(agm_sample, sample_labels)],
        #     index=dataframe.columns)
        #  dataframe = dataframe.append(to_append, ignore_index=True)
        #  dataframe.to_csv('sbm_agm_ranges.csv')


        stats = agm.statistics(agm_sample, sample_labels)
        G = agm.making_graph(agm_sample, sample_labels)


        out_pars = self.chars_to_array(stats)
        loss = self.loss_func(out_pars, self.targets)
        nums_to_del = []

        for num, targets_check in enumerate(self.characteristics_check):
            name = "".join(list(map(lambda x: str(x), targets_check[:-1])))
            if os.path.exists("../dataset/graph_" + str(name) + ".pickle"):
                nums_to_del.append(num)
            else:
                diff = np.abs((out_pars - targets_check))
                if np.all(diff < self.chars_to_array(self.limits)):
                    to_append = list(out_pars)
                    row_series = pd.Series(to_append, index=self.df_bench.columns)
                    self.df_bench = self.df_bench.append(row_series, ignore_index=True)


                    labels = []
                    for i, lab in G.nodes("label"):
                        labels.append(lab)
                    labels = np.array(labels)

                    np.save(
                        "../dataset/graph_" + str(name) + "_edgelist.npy",
                        np.array(G.edges()),
                    )

                    with open("../dataset/graph_" + str(name) + ".pickle", "wb") as f:
                        pickle.dump(G, f)

                    np.save("../dataset/graph_" + str(name) + "_labels.npy", labels)
                    nums_to_del.append(num)

        for index in sorted(nums_to_del, reverse=True):
            del self.characteristics_check[index]
        resp = self.array_to_chars(out_pars)
        resp["loss"] = loss.sum()
        resp["status"] = STATUS_OK
        return resp

    def run(self):
        """
        Run this class optimizing parameters for graph characteristics
        """
        for targets in self.characteristics:
            print(targets)
            self.targets = targets

            for tr in self.trials.trials:
                par = self.chars_to_array(tr["result"])
                if par is not None:
                    tr["result"]["loss"] = self.loss_func(par, targets)

            self.max_eval = len(self.trials.trials) + self.number_of_trials
            best = fmin(
                self.objective,
                self.hp_space,
                trials=self.trials,
                algo=atpe.suggest,
                max_evals=self.max_eval,
                early_stop_fn=self.early_stop,
            )
