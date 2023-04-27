import pygad
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

N = 100
M = 500
G = nx.gnm_random_graph(N, M)
print("density", nx.density(G))
print("clustering before", nx.average_clustering(G))
print("average degree", np.mean(list(zip(*nx.degree(G)))[1]))

desired_cl = 0.3
initial_population = list(map(lambda x: list(x), G.edges()))


desired_deg = 20



def fitness_func(instance, edge, idx):
    #  print(edges)
    # g = nx.from_edgelist(edges)
    # output = nx.average_clustering(g)
    population = instance.population
    # output1 = len(list(filter(lambda x: x[0]==edge[0] or x[1]==edge[0], population)))
    # output2 = len(list(filter(lambda x: x[0] == edge[0] or x[1] == edge[0], population)))
    # output=(output1+output2)*0.5


    avg_degs = []
    for i in range(N):
        avg_degs.append(len(list(filter(lambda x: x[0] == i or x[1] == i, population))))

    output = np.mean(avg_degs)

    # print(list(filter(lambda x: x[0]==edges[0] or x[1]==edges[0], initial_population)))
    eps = 10e-5
    fitness = 1.0 / (np.abs(output - desired_deg) + eps)
    # print(fitness)
    return fitness



fitness_function = fitness_func

num_generations = 200
num_parents_mating = 500

sol_per_pop = 1
num_genes = 2

init_range_low = 0
init_range_high = 99

parent_selection_type = "sus"
keep_parents = -1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 20
print(initial_population)



def on_generation(ga):
    print("Generation", ga.generations_completed)
    # print((ga.population))
    g = nx.from_edgelist(ga.population)
    nx.draw(g, pos=nx.spring_layout(g))
    plt.show()

    # avg_degs = []
    #  for i in range(N):
    #       avg_degs.append(len(list(filter(lambda x: x[0] == i or x[1] == i, ga.population))))

    #    output = np.mean(avg_degs)

    print(ga.best_solution()[1])


ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_function,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    parent_selection_type=parent_selection_type,
    keep_parents=keep_parents,
    crossover_type=crossover_type,
    gene_type=int,
    initial_population=initial_population,
    mutation_type=mutation_type,
    mutation_percent_genes=mutation_percent_genes,
    on_generation=on_generation,
)


ga_instance.run()

edges = ga_instance.population
G_after = nx.from_edgelist(edges)

print("population after", edges)
print(
    "after",
    nx.average_clustering(G_after),
    np.mean(list(zip(*nx.degree(G)))[1]),
    nx.number_connected_components(G_after),
)
# print("Parameters of the best solution : {solution}".format(solution=edges))
# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

# g = nx.from_edgelist(edges)
# prediction = nx.average_clustering(g)

