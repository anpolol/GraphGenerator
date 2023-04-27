import sys, math, random, copy, matplotlib.pyplot as plt
import collections
import numpy as np
import networkx as nx
import igraph as ig

# Just to double check our graph stats look pretty good
import pandas as pd
from scipy.stats import betabinom, rv_discrete

def ComputeDegreeDistribution(network):
	degrees = []
	for vertex, neighbors in network.items():
		degrees.append(len(neighbors))
	degrees.sort()
	vals = range(len(degrees))
	vals = list(map(float, vals))
	vals = list(map(lambda x: 1 - x / (len(degrees)-1), vals))
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

	for vertex,neighbors in network.items():
		for neighbor in neighbors:
			std1 += (labels[vertex]-mean1)**2
			std2 += (labels[neighbor]-mean2)**2
			cov += (labels[vertex]-mean1)*(labels[neighbor]-mean2)

	std1 = math.sqrt(std1)
	std2 = math.sqrt(std2)
	return cov / (std1*std2)

class PreferentialAttachment:
	def __init__(self, N, k0, k):
		self.N = N
		self.k0 = k0 #изначальное число вершин
		self.k = k #сколько связей на каждом шаге добавляется

	def Sample(self):
		degree_dist = []
		for i in range(int(self.k0)):
			degree_dist.extend([i]*int((self.k0-1)))

		for i in range(self.k0, self.N):
			newvals = []
			for j in range(self.k):
				newvals.append(random.choice(degree_dist))
			degree_dist.extend(newvals)
			degree_dist.extend([i]*self.k)
		return range(self.N), degree_dist


class PowerLaw:
	def __init__(self, N, min_d, max_d,power):
		self.N = N
		self.min_d = min_d #изначальное число вершин
		self.max_d = max_d #сколько связей на каждом шаге добавляется
		self.power = power

	def sum(self, min_d: int, max_d: int):
		"""
        Calculate the sum of inverse power for power degree distribution

        :param min_d: (int): Degree value of the node with the minimum degree
        :param max_d: (int): Degree value of the node with the maximum degree
        :return: (float): The sum
        """

		sum = float(np.sum([1 / (pow(i, 2.0)) for i in range(min_d, max_d + 1)]))

		return sum

	def pk(self, min_d: int, max_d: int,power:float):
		"""
        Build a power distribution of degrees

        :param min_d: (int): Degree value of the node with the minimum degree
        :param max_d: (int): Degree value of the node with the maximum degree
        :return: (Tuple[float]): The power degree distribution
        """
		probs = []
		sum = self.sum(min_d, max_d)

		for x in range(min_d, max_d + 1):
			probability = 1 / (pow(x, power) * sum)
			probs.append(probability)
		return tuple(probs)

	def Sample(self):

		rand_power_law = rv_discrete(
			self.min_d, self.max_d, values=(range(self.min_d, self.max_d + 1), self.pk(self.min_d, self.max_d, self.power))
		)
		degrees = np.sort(rand_power_law.rvs(size=self.N))

		degree_dist = []
		for i,deg in enumerate(degrees):
			degree_dist.extend([i]*deg)
		return range(self.N), degree_dist

# The FCL sampler we'll use for a proposal distribution
class FastChungLu:
	def __init__(self, vertices, degree_dist):
		self.vertex_list = vertices
		self.degree_distribution = degree_dist

	def sample_edge(self):
		vertex1 = self.degree_distribution[random.randint(0,len(self.degree_distribution)-1)]
		vertex2 = self.degree_distribution[random.randint(0,len(self.degree_distribution)-1)]
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
				ne+=2

		return sample_network


# The FCL sampler we'll use for a proposal distribution
class SBM:
	def __init__(self, vertices, degree_dist,number_of_groups,probs,size):
		self.vertex_list = vertices
		self.degree_distribution = degree_dist
		self.number_of_groups = number_of_groups
		self.probs = probs
		self.size = size
		groups = np.random.randint(0, self.number_of_groups , self.size)
		self.n_to_groups = dict(list(zip(np.arange(self.size), groups)))


	def sample_edge(self):
		vertex1 = self.degree_distribution[random.randint(0,len(self.degree_distribution)-1)]

		degrees = np.array(list(collections.Counter(self.degree_distribution).values()))
		probs_for_v1 = []
		for node, deg in enumerate(degrees):
			probs_for_v1.append(self.probs[self.n_to_groups[vertex1]][self.n_to_groups[node]])
		weights = degrees * np.array(probs_for_v1)

		vertex2 = random.choices(list(range(self.size)),weights=weights,k=1)#self.degree_distribution[random.randint(0,len(self.degree_distribution)-1)]
		return vertex1, vertex2[0]

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
				ne+=2

		return sample_network

# The AGM process.  Overall, most of the work is done in either the edge_acceptor or the proposing distribution
class AGM:
	# Need to keep track of how many edges to sample
	def __init__(self, degrees_dist, assort_in,assort_between,L, sample_network,alpha,beta):
		self.ne = 0
		self.assort_in = assort_in
		self.assort_out = assort_between
		self.L= L

		degrees = list(collections.Counter(degrees_dist).values())

		for deg in degrees:
			self.ne += deg

		self.acceptence_probability = self.acceptence_probability_count(sample_network,alpha,beta)

	def acceptence_probability_count(self, sample_networks, alpha,beta):


		true_counts = {}
		true_probs = {}
		sample_counts = {}
		sample_probs = {}
		self.ratios = {}
		self.acceptance_probs = {}


		attributes = {}
		for vertex in sample_networks:
			attr = np.random.rand(16)
			attr = attr / np.linalg.norm(attr)
			attributes[vertex] = attr

		ne = 0
		# Determine the attribute distribution in the sampled network
		for i in range(11):
			sample_counts[i] = 1
			true_counts[i] = 1
		for vertex, neighbors in sample_networks.items():
			for neighbor in neighbors:
				ne+=1
				attr1 = attributes[vertex]
				attr2=attributes[neighbor]
				var = round(10*(np.dot(attr1,attr2)+1)/2)
				sample_counts[var] += 1.0

		print(sample_counts)
		total = sum(sample_counts.values())
		for val, count in sample_counts.items():
			sample_probs[val] = count / total

		rv = betabinom(10, alpha, beta)
		true_counts = dict(collections.Counter(rv.rvs(ne)))
		total = sum(true_counts.values())
		for val, count in true_counts.items():
			true_probs[val] = count / total
		# Create the ratio between the true values and sampled values

		print(true_counts)

		for val in true_counts.keys():
			self.ratios[val] = true_probs[val] / sample_probs[val]

		# Normalize to figure out the acceptance probabilities
		max_val = max(self.ratios.values())
		for val, ratio in self.ratios.items():
			self.acceptance_probs[val] = ratio / max_val

	def accept_or_reject(self, label1, label2,assort_in,assort_out):
		if label1==label2:
			if random.random() < assort_in:
				return True
		elif label1!=label2:
			if random.random() < assort_out/self.L:
				return True

	def accept_or_reject_attributes(self, attr1, attr2):
		var = round(10*(np.dot(attr1,attr2)+1)/2)
		if random.random() < self.acceptance_probs[var]:
			return True

		return False


	# Create a new graph sample
	def sample_graph(self, proposal_distribution, labels):
		sample_network = {}
		for vertex in proposal_distribution.vertex_list:
			sample_network[vertex] = {}

		sampled_ne = 0
		attributes = {}
		while sampled_ne < self.ne:
			v1, v2 = proposal_distribution.sample_edge()

			# The rejection step.  The first part is just making sure the edge doesn't already exist;
			# the second actually does the acceptance/not acceptance.  This requires the edge_accept
			# to have been previously trained
			if v2 not in sample_network[v1] and self.accept_or_reject(labels[v1], labels[v2], self.assort_in, self.assort_out):
				sample_network[v1][v2] = 1
				sample_network[v2][v1] = 1
				sampled_ne += 2

				to_change = []
				for v in [v1,v2]:
					if v not in attributes:
						attr = np.random.rand(16)
						attr = attr/np.linalg.norm(attr)
						attributes[v] = attr
						to_change.append(v)

				if len(to_change)>0:
					while not self.accept_or_reject_attributes(attributes[v1],attributes[v2]):
						for v in to_change:
							attr = np.random.rand(16)
							attr = attr / np.linalg.norm(attr)
							attributes[v] = attr



		return sample_network, attributes

	def statistics(self,sample_network,sample_labels,attributes):
		"""
        Calculate characteritics of the graph

        :return: ({Any: Any}): Dictionary of calculated graph characteristics in view 'name_of_characteristic:
        value_of_this_characteristic'
        """
		edges = []
		for node in sample_network:
			for neigh in sample_network[node]:
				edges.append([node,neigh])

		graph = nx.from_edgelist(edges)
		self.num_nodes = graph.number_of_nodes()

		dict_of_parameters = {
			"Avg Degree": np.mean(list(dict(graph.degree()).values())),
			"Cluster": nx.average_clustering(graph),
		}

		dict_of_parameters["Connected components"] = nx.number_connected_components(
			graph
		)

		if nx.number_connected_components(graph) == 1:
			iG = ig.Graph.from_networkx(graph)
			avg_shortest_path = 0
			for shortest_path in iG.distances():
				for sp in shortest_path:
					avg_shortest_path += sp
			avg_s_p = avg_shortest_path / (
					self.num_nodes * self.num_nodes - self.num_nodes
			)
		else:
			connected_components = dict_of_parameters["Connected components"]
			avg_shortes_path = 0
			for nodes in nx.connected_components(graph):
				g = graph.subgraph(nodes)
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

		feature_assort = []
		for node in sample_network:
			s = 0
			for neigh in sample_network[node]:
				a = attributes[node]
				b = attributes[neigh]
				cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
				if cos_sim>0.7:
					s+=1
			feature_assort.append(s/len(sample_network[node]))

		dict_of_parameters["Avg shortest path"] = avg_s_p
		dict_of_parameters['assortativity'] = ComputeLabelCorrelation(sample_network, sample_labels)
		dict_of_parameters['Feature Assort'] = np.mean(feature_assort)

		return dict_of_parameters

	def making_graph(self,sample_network,sample_labels):
		G = nx.Graph()
		for i,label in enumerate(sample_labels):
			G.add_node(i,label=label)
		for v1 in sample_network:
			for v2 in sample_network[v1]:
				G.add_edge(v1,v2)
		return G

if __name__ == "__main__":
	# data location

	# Random permutation of labels.  This is shorter code than sampling bernoullis for all,
	# and can be replaced if particular labels should only exist with some guaranteed probability
	# for (e.g.) privacy
	dataframe = pd.DataFrame(
		columns=['Avg Degree', 'Cluster', 'Connected components', 'Avg shortest path', 'assortativity'])

	#параметры чтоб их тюнить
	N = 500 #size
	L = 10 #число меток в графе

	number_of_groups=3 #группы для SBM и метки - это разное.
	probs = [[0.8,0.3,0.2],[0.3,0.7,0.4],[0.2,0.4,0.9]]

	assort_corr_in = 0.8
	assort_corr_between = 0.2

	#параметры механизма Preferencial attachment  - вероятно буду менять механизм через randPL
	k0 = 5
	k=5
	b = 0
	alpha = 5.5
	beta = 1.5
	min_d = 5
	max_d = 100
	power = 2
	for number_of_groups in [3]:
		for inside_prob in [0.7]:
			for outside_prob in [0.3]:
				probs = np.diag(k*[inside_prob])
				probs = np.where(probs==0, inside_prob,outside_prob)
				probs = probs.tolist()
				for assort_corr_in in [0.5]:
					for assort_corr_between in [0.2]:
							print(b)
							b+=1
							sample_labels_keys = list(range(N))
							sample_labels_items = random.choices(list(range(L)), k=N)
							sample_labels = dict(zip(sample_labels_keys, sample_labels_items))

							# Now for the AGM steps.  First, just create the AR method using the given data, the proposing distribution,
							# and the random sample of neighbors.

							# Now we actually do AGM!  Just plug in your proposing distribution (FCL Example Given) as well as
							# the edge_acceptor, and let it go!
						#	pa = PreferentialAttachment(N, k0, k)
							pa = PowerLaw(N,min_d,max_d,power)
							vertices, degree_dist = pa.Sample()
							#generator = FastChungLu(vertices, degree_dist)
							generator = SBM(vertices,degree_dist,number_of_groups,probs,N)
							sample_network = generator.sample_graph()
							agm = AGM(degree_dist,assort_corr_in,assort_corr_between,L,sample_network,alpha,beta)
							agm_sample, attributes = agm.sample_graph(generator, sample_labels)



							dict_statistics = agm.statistics(agm_sample,sample_labels,attributes)
							# This should be fairly close to the initial graph's correlation
							print('assort', dict_statistics['Feature Assort'])


	#xs, ys = ComputeDegreeDistribution(agm_sample)
	#plt.plot(xs, ys, label='AGM-FCL')
	#plt.legend()
	#plt.xscale('log')
	#plt.yscale('log')
	#plt.xlabel('Degree')

	#plt.ylabel('CCDF')
	#plt.show()
