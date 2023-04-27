import copy
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, atpe, fmin, hp, tpe
from numpy.typing import ArrayLike

from core.attributed_generator import AttributedGenerator as Model


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
            "sigma_init": hp.uniform("sigma_init", 0.8, 1.1),
            "power": hp.uniform("power", 2, 3),
            "max_d": hp.quniform("max_d", 300, 1000, 100),
            "num_classes": hp.quniform("num_classes", 5, 20, 5),
            "etta": hp.uniform("etta", 0, 5),
            "ro": hp.uniform("ro", 0, 1),
            "mu": hp.uniform("mu", 0.1, 0.95),
            "min_d": hp.quniform("min_d", 1, 10, 1),
        }
        self.OUT_PAR_NAMES = [
            "Label Assort",
            "Feature Assort",
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
            "Label Assort": 0.3,
            "Feature Assort": 0.3,
            "Cluster": 0.05,
            "Avg shortest path": 0.3,
            "Avg Degree": 1.5,
            "Connected components": 10,
        }
        self.df_bench = pd.DataFrame(
            columns=[
                "Label assort",
                "Feature assort",
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
        model = Model(
            num_nodes=1000,
            max_d=int(args["max_d"]),
            num_classes=int(args["num_classes"]),
            etta=args["etta"],
            ro=args["ro"],
            mu=args["mu"],
            sigma_init=args["sigma_init"],
            sigma_every=1,
            dim=128,
            power=args["power"],
            min_d=int(args["min_d"]),
        )

        G, _ = model.generate()

        stats = model.statistics()
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

                    node_attr = []
                    for i, attr in G.nodes("attribute"):
                        node_attr.append(attr.tolist())
                    labels = []
                    for i, lab in G.nodes("label"):
                        labels.append(lab)
                    labels = np.array(labels)

                    np.save(
                        "../dataset/graph_" + str(name) + "_edgelist.npy",
                        np.array(G.edges()),
                    )
                    np.save(
                        "../dataset/graph_" + str(name) + "_attr.npy",
                        node_attr,
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
