import random

import networkx as nx
import numpy as np

import warnings

# Ignore runtime warning message
warnings.simplefilter("ignore", category=RuntimeWarning, append=True)


# functia de fitness
def fitness_function(chromosome, network):
    # calculate modularity for given chromosome
    m = network["noEdges"]
    Q = 0
    for i in range(len(chromosome)):
        for j in range(len(chromosome)):
            if chromosome[i] == chromosome[j]:
                Aij = network["mat"][i][j]
                ki = network["degrees"][i]
                kj = network["degrees"][j]
                Q += (Aij - (ki * kj) / (2 * m))
    return Q / (4 * m)


# genetic operators
def mutation(chromosome, mutation_prob):
    # we do a mutation with a certain probability
    if np.random.rand() < mutation_prob:
        idx = np.random.randint(len(chromosome))
        chromosome[idx] = np.random.randint(max(chromosome) + 1)
    return chromosome


def crossover(chromosome1, chromosome2, crossover_point):
    # we cross two chromose at the given crossover point
    new_chromosome1 = np.concatenate([chromosome1[:crossover_point], chromosome2[crossover_point:]])
    new_chromosome2 = np.concatenate([chromosome2[:crossover_point], chromosome1[crossover_point:]])
    return new_chromosome1, new_chromosome2


def selection(population, fitness_values):
    # # we select 2 candidates to be passed forward to the next generation

    shifted_fitness_values = fitness_values - np.min(fitness_values)
    # Normalize shifted fitness values
    fitness_values_normalized = shifted_fitness_values / np.sum(shifted_fitness_values)
    # print("Fitness values normalized: ", fitness_values_normalized)
    fitness_values_normalized = np.nan_to_num(fitness_values_normalized)
    # Select two individuals using the normalized fitness values
    idx1 = np.random.choice(range(len(population)), p=fitness_values_normalized)
    idx2 = np.random.choice(range(len(population)), p=fitness_values_normalized)

    return population[idx1], population[idx2]


# genetic algorithm
def genetic_algorithm(network, population_size, num_generations, mutation_prob, crossover_point, num_communities):
    # we randomly initialize the populations with chromosomes
    population = np.random.randint(0, num_communities, size=(population_size, network["noNodes"]))
    index = 1
    for generation in range(num_generations):
        # evaluate fitness for every chromosome
        fitness_values = np.array([fitness_function(chromosome, network) for chromosome in population])
        shifted_fitness_values = fitness_values - np.min(fitness_values)
        fitness_values_normalized = shifted_fitness_values / np.sum(shifted_fitness_values)
        if (np.isclose(sum(fitness_values_normalized), 1)):
            # we select the two best candidates and we pass the forward to the next generation
            new_population = []
            for _ in range(population_size):
                chromosome1, chromosome2 = selection(population, fitness_values)
                new_chromosome1, new_chromosome2 = crossover(chromosome1, chromosome2, crossover_point)
                new_population.append(mutation(new_chromosome1, mutation_prob))
                new_population.append(mutation(new_chromosome2, mutation_prob))

            population = np.array(new_population)
            index = index + 1
        else:
            fitness_values = np.array([fitness_function(chromosome, network) for chromosome in population])
            best_chromosome = population[np.argmax(fitness_values)]
            print("\nbest chromosome: ", best_chromosome)
            return best_chromosome

    # find the best chromosome and return it
    fitness_values = np.array([fitness_function(chromosome, network) for chromosome in population])
    best_chromosome = population[np.argmax(fitness_values)]
    print("best chromosome: ", best_chromosome)
    return best_chromosome


def readNet(fileName):
    f = open(fileName, "r")
    net = {}
    n = int(f.readline())
    net['noNodes'] = n
    mat = []
    for i in range(n):
        mat.append([])
        line = f.readline()
        elems = line.split(" ")
        for j in range(n):
            mat[-1].append(int(elems[j]))
    net["mat"] = mat
    degrees = []
    noEdges = 0
    for i in range(n):
        d = 0
        for j in range(n):
            if mat[i][j] == 1:
                d += 1
            if j > i:
                noEdges += mat[i][j]
        degrees.append(d)
    net["noEdges"] = noEdges
    net["degrees"] = degrees
    communities = []
    for i in range(n):
        communities.append([i + 1])
    net["communities"] = communities
    f.close()
    return net


def readGML(filename):
    G = nx.read_gml(filename, label='id')
    print(G)
    # print(nx.to_numpy_array(G))
    net = {"mat": nx.to_numpy_array(G)}
    communities = []
    for i in range(nx.number_of_nodes(G)):
        communities.append([i + 1])
    net["communities"] = communities
    net["noNodes"] = nx.number_of_nodes(G)
    net["noEdges"] = nx.number_of_edges(G)
    degrees = []
    for i in range(net["noNodes"]):
        d = 0
        for j in range(net["noNodes"]):
            if net["mat"][i][j] == 1:
                d += 1
        degrees.append(d)
    net["degrees"] = degrees
    # print(net)
    return net


def writeChromosomeInFile(chromosome, fileName):
    file = open(fileName, "w")
    index = 1
    for value in chromosome[:-1]:
        file.write(f" {index} {value + 1}\n")
        index = index + 1
    file.write(f" {index} {chromosome[-1] + 1}")
    file.close()


def getMatchingPercentageOfTwoFiles(fileName1, fileName2):
    with open(fileName1, 'r') as file1, open(fileName2, 'r') as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()
    index = 0
    total = 0
    matches = 0
    for line in lines1:
        int_list1 = [int(x) for x in line.split()]
        int_list2 = [int(x) for x in lines2[index].split()]
        index = index + 1
        j = 0
        for nr in int_list1:
            if nr == int_list2[j]:
                matches = matches + 1
            j = j + 1
            total = total + 1
    percentage = matches / total*100
    print(f"Matching percentage: {percentage:.2f}%")


def main():
    # print("result for net.in:")
    # print(readNet("net.in"))
    # print(genetic_algorithm(readNet("net.in"),10,6,0.01,3,2))

    print("result for dolphins.gml:")
    writeChromosomeInFile(
        genetic_algorithm(readGML("real/dolphins/dolphins.gml"),150,200,0.02,random.randint(1,62),2),
        "results/dolphins.txt")
    getMatchingPercentageOfTwoFiles("results/dolphins.txt","real/dolphins/classLabeldolphins.txt")

    print("\nresult for karate.gml:")
    writeChromosomeInFile(
        genetic_algorithm(readGML("real/karate/karate.gml"),150,200,0.01,random.randint(1,34),2),
        "results/karate.txt")
    getMatchingPercentageOfTwoFiles("results/karate.txt","real/karate/classLabelkarate.txt")

    print("\nresult for krebs.gml:")
    writeChromosomeInFile(
        genetic_algorithm(readGML("real/krebs/krebs.gml"),150,200,0.01,random.randint(1,104),3),
        "results/krebs.txt")
    getMatchingPercentageOfTwoFiles("results/krebs.txt","real/krebs/classLabelkrebs.txt")

    print("\nresult for football.gml:")
    writeChromosomeInFile(
        genetic_algorithm(readGML("real/football/football.gml"), 100, 300, 0.01, random.randint(1, 114), 12),
        "results/football.txt")
    getMatchingPercentageOfTwoFiles("results/football.txt", "real/football/classLabelfootball.txt")

    print("\nresult for data1.in: ")
    writeChromosomeInFile(
        genetic_algorithm(readNet("real/dataMadeByME/data1.in"),40,40,0.01,random.randint(1,7),2),"results/data1.out"
    )

    print("\nresult for data2.in: ")
    writeChromosomeInFile(
        genetic_algorithm(readNet("real/dataMadeByME/data2.in"), 40, 40, 0.01, random.randint(1, 8), 2),
        "results/data2.out"
    )

    print("\nresult for data3.in: ")
    writeChromosomeInFile(
        genetic_algorithm(readNet("real/dataMadeByME/data3.in"), 40, 40, 0.01, random.randint(1, 7), 2),
        "results/data3.out"
    )

    print("\nresult for data4.in: ")
    writeChromosomeInFile(
        genetic_algorithm(readNet("real/dataMadeByME/data4.in"), 40, 40, 0.01, random.randint(1, 7), 2),
        "results/data4.out"
    )

    print("\nresult for data5.in: ")
    writeChromosomeInFile(
        genetic_algorithm(readNet("real/dataMadeByME/data5.in"), 40, 40, 0.01, random.randint(1, 6), 2),
        "results/data5.out"
    )

    print("\nresult for data6.in: ")
    writeChromosomeInFile(
        genetic_algorithm(readNet("real/dataMadeByME/data6.in"), 40, 40, 0.01, random.randint(1, 7), 2),
        "results/data6.out"
    )


main()
