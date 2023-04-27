import sys, math, random, copy, matplotlib.pyplot as plt
import collections
import numpy as np
import networkx as nx
import igraph as ig

# Just to double check our graph stats look pretty good
import pandas as pd

from scipy.stats import beta, betabinom


def ComputeDegreeDistribution(network):
    degrees = []
    for vertex, neighbors in network.items():
        degrees.append(len(neighbors))

    degrees.sort()

    vals = range(len(degrees))
    vals = list(map(float, vals))
    vals = list(map(lambda x: 1 - x / (len(degrees) - 1), vals))

    return vals, degrees


# Just computes the pearson correlation
def ComputeLabelCorrelation(network, labels):
    mean1 = 0.0
    mean2 = 0.0
    total = 0.0
    for vertex, neighbors in network.items():
        for neighbor in neighbors:
            mean1 += labels[vertex]
            mean2 += labels[neighbor]
            total += 1

    mean1 /= total
    mean2 /= total
    std1 = 0.0
    std2 = 0.0
    cov = 0.0

    for vertex, neighbors in network.items():
        for neighbor in neighbors:
            std1 += (labels[vertex] - mean1) ** 2
            std2 += (labels[neighbor] - mean2) ** 2
            cov += (labels[vertex] - mean1) * (labels[neighbor] - mean2)

    std1 = math.sqrt(std1)
    std2 = math.sqrt(std2)
    return cov / (std1 * std2)


class PreferentialAttachment:
    def __init__(self, N, k0, k):
        self.N = N
        self.k0 = k0  # изначальное число вершин
        self.k = k  # сколько связей на каждом шаге добавляется

    def Sample(self):
        degree_dist = []
        for i in range(int(self.k0)):
            degree_dist.extend([i] * int((self.k0 - 1)))

        for i in range(self.k0, self.N):
            newvals = []
            for j in range(self.k):
                newvals.append(random.choice(degree_dist))
            degree_dist.extend(newvals)
            degree_dist.extend([i] * self.k)
        return range(self.N), degree_dist


# The FCL sampler we'll use for a proposal distribution
class FastChungLu:
    def __init__(self, network):
        self.vertex_list = []
        self.degree_distribution = []
        for vertex, neighbors in network.items():
            self.vertex_list.append(vertex)
            self.degree_distribution.extend([vertex] * len(neighbors))

    def sample_edge(self):
        vertex1 = self.degree_distribution[
            random.randint(0, len(self.degree_distribution) - 1)
        ]
        vertex2 = self.degree_distribution[
            random.randint(0, len(self.degree_distribution) - 1)
        ]

        return vertex1, vertex2

    def sample_graph(self):
        sample_network = {}
        for vertex in self.vertex_list:
            sample_network[vertex] = {}

        ne = 0
        while ne < len(self.degree_distribution):
            v1, v2 = self.sample_edge()
            if v2 not in sample_network[v1]:
                sample_network[v1][v2] = 1
                sample_network[v2][v1] = 1
                ne += 2

        return sample_network


# A simple A/R that creates the following edge features from the corresponding vertex
# attributes.  Namely, if both are 0, if both are 1, and if both are 2.
class SimpleBernoulliAR:
    # Returns 0/0 -> 0, 0/1->1, 1/0->1, 1/1 -> 2
    def edge_var(self, label1, label2):
        return label1 * label1 + label2 * label2

    # Requires the true network, a complete sampled network from the proposing distribution
    # then the true labels and a random sample of labels
    def learn_ar(self, true_network, sampled_network, true_labels, sample_labels):
        true_counts = {}
        true_probs = {}
        sample_counts = {}
        sample_probs = {}
        self.ratios = {}
        self.acceptance_probs = {}
        print("true_labels", collections.Counter(list(true_labels.values())).keys())
        # Determine the attribute distribution in the real network
        for vertex, neighbors in true_network.items():
            for neighbor in neighbors:
                var = self.edge_var(true_labels[vertex], true_labels[neighbor])
                if var not in true_counts:
                    # put a small (dirichlet) prior
                    true_counts[var] = 1.0
                true_counts[var] += 1
        total = sum(true_counts.values())
        for val, count in true_counts.items():
            true_probs[val] = count / total
        print("true probs", collections.Counter(list(true_labels.keys())).keys())

        # Determine the attribute distribution in the sampled network
        for vertex, neighbors in sampled_network.items():
            for neighbor in neighbors:
                var = self.edge_var(sample_labels[vertex], sample_labels[neighbor])
                if var not in sample_counts:
                    # put a small (dirichlet) prior
                    sample_counts[var] = 1.0
                sample_counts[var] += 1.0
        total = sum(sample_counts.values())
        for val, count in sample_counts.items():
            sample_probs[val] = count / total

        # Create the ratio between the true values and sampled values
        for val in true_counts.keys():
            self.ratios[val] = true_probs[val] / sample_probs[val]

        # Normalize to figure out the acceptance probabilities
        max_val = max(self.ratios.values())
        for val, ratio in self.ratios.items():
            self.acceptance_probs[val] = ratio / max_val

    def accept_or_reject(self, label1, label2, assort_in, assort_out, my):
        if my:
            if label1 == label2:
                if random.random() < assort_in:
                    return True
            elif label1 != label2:
                if random.random() < assort_out / 20:
                    return True
        else:
            if random.random() < self.acceptance_probs[self.edge_var(label1, label2)]:
                return True

        return False


# The AGM process.  Overall, most of the work is done in either the edge_acceptor or the proposing distribution
class AGM:
    # Need to keep track of how many edges to sample
    def __init__(self, network, assort_in, assort_between, my):
        self.ne = 0
        self.assort_in = assort_in
        self.assort_out = assort_between
        self.my = my

        for vertex, neighbors in network.items():
            self.ne += len(neighbors)

    # Create a new graph sample
    def sample_graph(self, proposal_distribution, labels, edge_acceptor):
        sample_network = {}
        for vertex in proposal_distribution.vertex_list:
            sample_network[vertex] = {}

        sampled_ne = 0
        while sampled_ne < self.ne:
            v1, v2 = proposal_distribution.sample_edge()

            # The rejection step.  The first part is just making sure the edge doesn't already exist;
            # the second actually does the acceptance/not acceptance.  This requires the edge_accept
            # to have been previously trained
            if v2 not in sample_network[v1] and edge_acceptor.accept_or_reject(
                labels[v1], labels[v2], self.assort_in, self.assort_out, self.my
            ):
                sample_network[v1][v2] = 1
                sample_network[v2][v1] = 1
                sampled_ne += 2

        return sample_network


if __name__ == "__main__":
    # data location
    data = "test"
    data_dir = ""

    # edge representation
    network = {}
    # corresponding labels
    labels = {}

    # reading the edge file
    with open(data_dir + data + ".edges") as edge_file:
        for line in edge_file:
            fields = list(map(int, line.strip().split("::")))

            # ids
            id0 = fields[0]
            id1 = fields[1]

            # Remove self-loops
            if id0 == id1:
                continue

            # Check/create new vertices
            if id0 not in network:
                network[id0] = {}
            if id1 not in network:
                network[id1] = {}

            network[id0][id1] = 1
            network[id1][id0] = 1

    # readin the label file
    with open(data_dir + data + ".lab") as label_file:
        for line in label_file:
            fields = list(map(int, line.strip().split("::")))

            # values
            id = fields[0]
            lab = fields[1]

            # only include items with edges
            if id not in network:
                continue

            # assign the labels
            labels[id] = lab

    print("Initial Graph Correlation", ComputeLabelCorrelation(network, labels))
    fcl = FastChungLu(network)
    fcl_sample = fcl.sample_graph()

    # Random permutation of labels.  This is shorter code than sampling bernoullis for all,
    # and can be replaced if particular labels should only exist with some guaranteed probability
    # for (e.g.) privacy
    sample_labels_keys = copy.deepcopy(list(labels.keys()))
    sample_labels_items = copy.deepcopy(list(labels.values()))
    random.shuffle(sample_labels_items)
    sample_labels = dict(zip(sample_labels_keys, sample_labels_items))

    # Double check that the FCL correlation is negligible (if this is not near 0 there's something wrong)
    print("FCL Graph Correlation", ComputeLabelCorrelation(fcl_sample, sample_labels))

    # Now for the AGM steps.  First, just create the AR method using the given data, the proposing distribution,
    # and the random sample of neighbors.
    edge_acceptor = SimpleBernoulliAR()
    edge_acceptor.learn_ar(network, fcl_sample, labels, sample_labels)

    # Now we actually do AGM!  Just plug in your proposing distribution (FCL Example Given) as well as
    # the edge_acceptor, and let it go!
    agm = AGM(network, 0.8, 0.3, True)
    agm_sample = agm.sample_graph(fcl, sample_labels, edge_acceptor)

    # This should be fairly close to the initial graph's correlation
    print("AGM Graph Correlation", ComputeLabelCorrelation(agm_sample, sample_labels))

    xs, ys = ComputeDegreeDistribution(network)
    plt.plot(xs, ys, label="Original")
    xs, ys = ComputeDegreeDistribution(fcl_sample)
    plt.plot(xs, ys, label="FCL")
    xs, ys = ComputeDegreeDistribution(agm_sample)
    plt.plot(xs, ys, label="AGM-FCL")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree")
    # plt.ylabel('CCDF')
    plt.show()
