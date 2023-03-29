import random
import os
import networkx as nx
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
from src.graph_tool.generation import generate_sbm
#print(os._version_)
sizes = [75, 75, 100]
probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]

cluster = []
avgd = []
asps = []

n = 10
b = random.choices([0,1], k=n)
degs = [10,9,8,7,6,2,2,1,1,1]
out_degs = list(np.array(degs)*0.1)
print(out_degs)
in_degs = list(np.array(degs)-np.array(out_degs))
print(in_degs)

sbm = generate_sbm(b = b, out_degs = out_degs, in_degs = in_degs, micro_degs=True)
print(sbm.degree_property_map())

for i in range(1):
    #print(i)
    g = nx.stochastic_block_model(sizes, probs)
    #cluster.append(nx.average_clustering(g))
   # asps.append(nx.average_shortest_path_length(g), )
  #  print(np.mean(list(zip(*g.degree()))[1]))
    avgd.append(np.mean(list(zip(*g.degree()))[1]))
    (sns.histplot(list(zip(*g.degree()))[1]))
    plt.show()
print(np.mean(avgd), np.std(avgd))