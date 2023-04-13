# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# 1/15/15 -- author: Joel Pfeiffer.  jpfeiffer@purdue.edu
# Simple demonstration of the AGM sampler in conjunction with
# the FCL proposal distribution
#
# This method utilizes two papers: the first is the classic Barabasi-Albert model to create a degree distribution:
#
# Emergence of scaling in random networks
# Albert-Laszlo Barabasi and Reka Albert
# In Science 286 (5439): 509-512
#
# The second uses the degree distribution of the above paper to use in conjunction with AGM-FCL:
#
# Attributed Graph Models: Modeling network structure with correlated attributes
# Joseph J. Pfeiffer III, Sebastian Moreno, Timothy La Fond, Jennifer Neville and Brian Gallagher
# In Proceedings of the 23rd International World Wide Web Conference (WWW 2014), 2014
#
# Hence, the resulting network is a complete random sample with a random DD, as well as random
# label correlations across edges.  It also randomly samples conditional attributes.
#
# The intent of this work is to define a clear approach to sampling from distributions -- note,
# AGM can be considerably more complex.  Along these lines, most parameterizations are preset and
# a user might want to hand tune them in the code later.

import random, math, time, matplotlib.pyplot as plt


# Just to double check our graph stats look pretty good
def ComputeDegreeDistribution(network):
    degrees = []
    for vertex, neighbors in network.items():
        degrees.append(len(neighbors))

    degrees.sort()

    vals = range(len(degrees))
    vals = list(map(float, vals))
    vals = list(map(lambda x: 1 - x / (len(degrees) - 1), vals))

    return vals, degrees


def CreateLabelsAndAttr(
    network, prior=0.5, conds=[[0.3, 0.4, 0.35], [0.85, 0.65, 0.75]]
):  # тут учитывается корреляция меток и атрибутов
    labels = {}
    attrs = {}
    for vertex in network:
        labels[vertex] = int(random.random() < prior)
        attrs[vertex] = list(
            map(lambda x: int(random.random() < x), conds[labels[vertex]])
        )
    return labels, attrs


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


# Just computes the pearson correlation
def ComputeAttrCorrelation(network, attrs, ind=0):
    mean1 = 0.0
    mean2 = 0.0
    total = 0.0
    for vertex, neighbors in network.items():
        for neighbor in neighbors:
            mean1 += attrs[vertex][ind]
            mean2 += attrs[neighbor][ind]
            total += 1

    mean1 /= total
    mean2 /= total
    std1 = 0.0
    std2 = 0.0
    cov = 0.0

    for vertex, neighbors in network.items():
        for neighbor in neighbors:
            std1 += (attrs[vertex][ind] - mean1) ** 2
            std2 += (attrs[neighbor][ind] - mean2) ** 2
            cov += (attrs[vertex][ind] - mean1) * (attrs[neighbor][ind] - mean2)

    std1 = math.sqrt(std1)
    std2 = math.sqrt(std2)
    return cov / (std1 * std2)


# Just computes the pearson correlation
def ComputeLabelAttrCorrelation(labels, attrs, ind=0):
    mean1 = 0.0
    mean2 = 0.0
    total = 0.0
    for vertex, neighbors in labels.items():
        mean1 += labels[vertex]
        mean2 += attrs[vertex][ind]
        total += 1

    mean1 /= total
    mean2 /= total
    std1 = 0.0
    std2 = 0.0
    cov = 0.0

    for vertex, neighbors in labels.items():
        std1 += (labels[vertex] - mean1) ** 2
        std2 += (attrs[vertex][ind] - mean2) ** 2
        cov += (labels[vertex] - mean1) * (attrs[vertex][ind] - mean2)

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
        for i in range(self.k0):
            degree_dist.extend([i] * (self.k0 - 1))

        for i in range(self.k0, self.N):
            newvals = []
            for j in range(self.k):
                newvals.append(random.choice(degree_dist))
            degree_dist.extend(newvals)
            degree_dist.extend([i] * self.k)

        return range(self.N), degree_dist


# The FCL sampler we'll use for a proposal distribution
class FastChungLu:
    def __init__(self, vertices, degree_dist):
        self.vertex_list = vertices
        self.degree_distribution = degree_dist

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

    # Usual order -- 0/0: 1, 1/0: .75, 1/1:1
    def set_accept_probs(self, ap=[1.0, 0.8, 1.0]):
        self.acceptance_probs = ap
        return

    def accept_or_reject(self, label1, label2):
        if random.random() < self.acceptance_probs[self.edge_var(label1, label2)]:
            return True

        return False


# The AGM process.  Overall, most of the work is done in either the edge_acceptor or the proposing distribution
class AGM:
    # Need to keep track of how many edges to sample
    def __init__(self, network):
        self.ne = 0

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
                labels[v1], labels[v2]
            ):
                sample_network[v1][v2] = 1
                sample_network[v2][v1] = 1
                sampled_ne += 2

        return sample_network


def CreateAGMSamples(size, verbose_check=True, show_dd=False):
    # Create an initial degree distribution we can work with
    pa = PreferentialAttachment(size, 5, 5)
    vertices, degree_dist = pa.Sample()

    # Create an FCL graph from the initial degree distribution
    fcl = FastChungLu(vertices, degree_dist)
    fcl_sample = fcl.sample_graph()

    # Create some endpoint labels/attributes
    labels, attrs = CreateLabelsAndAttr(fcl_sample)

    # Now for the AGM steps.  First, just create the AR method using the given data, the proposing distribution,
    # and the random sample of neighbors.
    edge_acceptor = SimpleBernoulliAR()
    edge_acceptor.set_accept_probs()

    # Now we actually do AGM!  Just plug in your proposing distribution (FCL Example Given) as well as
    # the edge_acceptor, and let it go!
    agm = AGM(fcl_sample)
    agm_sample = agm.sample_graph(fcl, labels, edge_acceptor)

    # Check the correlations of the labels/attrs
    if verbose_check:
        print(
            "FCL Graph Label Correlation", ComputeLabelCorrelation(fcl_sample, labels)
        )
        print(
            "AGM Graph Label Correlation", ComputeLabelCorrelation(agm_sample, labels)
        )

    # Check the DD
    if show_dd:
        xs, ys = ComputeDegreeDistribution(fcl_sample)

        plt.plot(xs, ys, label="FCL")
        xs, ys = ComputeDegreeDistribution(agm_sample)
        plt.plot(xs, ys, label="AGM-FCL")
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Degree")
        plt.ylabel("CCDF")
        plt.show()

    return agm_sample, labels, attrs


if __name__ == "__main__":
    out = "./"
    # Create the following sizes
    sizes = [100, 1000]  # 10000,100000,1000000,10000000,100000000]
    # Number to create for each possible size
    count = 1
    # Give verbose checks of everything on every network
    verbose = True
    # Should we display the DD for each (not recommended if making a bunch)
    show_dd = True
    store = False
    timeit = True  # Time

    # Loop over all the sizes and the number we want
    for size in sizes:
        for c in range(count):
            print("Network Size:", size, " count:", c + 1, " of", count)

            # draw the sample
            t0 = time.time()
            agm_sample, labels, attrs = CreateAGMSamples(size, verbose, show_dd)
            t1 = time.time()
            if timeit:
                print("Time:", t1 - t0)
            if store:
                # Assigned name
                network_name = "network_" + str(size) + "_sample_" + str(c)

                with open(out + network_name + ".edges", "w") as f:
                    for vertex, neighbors in agm_sample.items():
                        for neighbor in neighbors:
                            f.write(str(vertex) + "::" + str(neighbor) + "\n")

                with open(out + network_name + ".lab", "w") as f:
                    for vertex, lab in labels.items():
                        f.write(str(vertex) + "::" + str(lab) + "\n")

                with open(out + network_name + ".attr", "w") as f:
                    for vertex, attr in attrs.items():
                        f.write(str(vertex))
                        for a in attr:
                            f.write("::" + str(a))
                        f.write("\n")
