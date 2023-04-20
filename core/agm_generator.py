import sys, math, random, copy, matplotlib.pyplot as plt
import collections
import numpy as np
import networkx as nx
import igraph as ig

# Just to double check our graph stats look pretty good
import pandas as pd


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
	def __init__(self, degrees_dist, assort_in,assort_between,L):
		self.ne = 0
		self.assort_in = assort_in
		self.assort_out = assort_between
		self.L= L

		degrees = list(collections.Counter(degrees_dist).values())

		for deg in degrees:
			self.ne += deg


	def accept_or_reject(self, label1, label2,assort_in,assort_out):
		if label1==label2:
			if random.random() < assort_in:
				return True
		elif label1!=label2:
			if random.random() < assort_out/self.L:
				return True


	# Create a new graph sample
	def sample_graph(self, proposal_distribution, labels):
		sample_network = {}
		for vertex in proposal_distribution.vertex_list:
			sample_network[vertex] = {}

		sampled_ne = 0
		while sampled_ne < self.ne:
			v1, v2 = proposal_distribution.sample_edge()

			# The rejection step.  The first part is just making sure the edge doesn't already exist;
			# the second actually does the acceptance/not acceptance.  This requires the edge_accept
			# to have been previously trained
			if v2 not in sample_network[v1] and self.accept_or_reject(labels[v1], labels[v2],self.assort_in,self.assort_out):
				sample_network[v1][v2] = 1
				sample_network[v2][v1] = 1
				sampled_ne += 2

		return sample_network

	def statistics(self,sample_network,sample_labels):
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

		dict_of_parameters["Avg shortest path"] = avg_s_p
		dict_of_parameters['assortativity'] = ComputeLabelCorrelation(sample_network, sample_labels)
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
	for number_of_groups in [2,3,4,5]:
		for inside_prob in [0.2,0.5,0.7]:
			for outside_prob in [0.3,0.6,0.8]:
				probs= np.diag(k*[inside_prob])
				probs=np.where(probs==0, inside_prob,outside_prob)
				probs=probs.tolist()
				for assort_corr_in in [0.1,0.5,0.9]:
					for assort_corr_between in [0.2,0.4,0.6,0.8]:
						for k in [3,5,10]:
							print(b)
							b+=1
							sample_labels_keys = list(range(N))
							sample_labels_items = random.choices(list(range(L)), k=N)
							sample_labels = dict(zip(sample_labels_keys, sample_labels_items))

							# Now for the AGM steps.  First, just create the AR method using the given data, the proposing distribution,
							# and the random sample of neighbors.

							# Now we actually do AGM!  Just plug in your proposing distribution (FCL Example Given) as well as
							# the edge_acceptor, and let it go!
							pa = PreferentialAttachment(N, k0, k)
							vertices, degree_dist = pa.Sample()
							#generator = FastChungLu(vertices, degree_dist)
							generator = SBM(vertices,degree_dist,number_of_groups,probs,N)

							agm = AGM(degree_dist,assort_corr_in,assort_corr_between,L)
							agm_sample = agm.sample_graph(generator, sample_labels)



							dict_statistics = agm.statistics(agm_sample,sample_labels)
							# This should be fairly close to the initial graph's correlation
							print('AGM Graph Correlation', ComputeLabelCorrelation(agm_sample, sample_labels))
							to_append = pd.Series([dict_statistics['Avg Degree'],dict_statistics['Cluster'],dict_statistics['Connected components'],dict_statistics['Avg shortest path'],ComputeLabelCorrelation(agm_sample, sample_labels)],index=dataframe.columns)
							dataframe=dataframe.append(to_append,ignore_index=True)
							dataframe.to_csv('sbm_agm_ranges.csv')

	#xs, ys = ComputeDegreeDistribution(agm_sample)
	#plt.plot(xs, ys, label='AGM-FCL')
	#plt.legend()
	#plt.xscale('log')
	#plt.yscale('log')
	#plt.xlabel('Degree')

	#plt.ylabel('CCDF')
	#plt.show()
