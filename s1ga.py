import numpy as np

# Parameters
sources = 2  # F1, F2
destinations = 3  # w1, w2, w3
modes = 2  # T1, T2
population_size = 10  # Number of chromosomes in the population
mutation = 0.1
gen = 5  # Number of generations

supply = [20, 30]
demand = [15, 25, 10]
capacity = [25, 35]

population = []
crossover = []
fitness_scores = []

lenpi = sources * destinations * modes
pi = list(range(lenpi))

def make_chromosome(temp_pi, templenpi, temp_supply, temp_demand, temp_capacity):   
    chromosome = np.zeros((sources, destinations, modes), dtype=int)
    temppi = temp_pi.copy()
    tempsupply = temp_supply[:]
    tempdemand = temp_demand[:]
    tempcapacity = temp_capacity[:]

    for z in range(templenpi):
        index = np.random.choice(temppi)
        temppi.remove(index)
        i = index // (destinations * modes)
        j = (index // modes) % destinations
        k = index % modes
        val = min(tempsupply[i], tempdemand[j], tempcapacity[k])
        chromosome[i, j, k] = val
        tempsupply[i] -= val
        tempdemand[j] -= val
        tempcapacity[k] -= val

    return chromosome

def make_crossover(chromo1, chromo2):
    count = 0
    t_chromo1 = np.copy(chromo1)
    t_chromo2 = np.copy(chromo2)

    for i in range(sources):
        for j in range(destinations):
            for k in range(modes):
                if np.random.rand() <= mutation:
                    t_chromo1[i, j, k] = np.random.randint(0, t_chromo1[i, j, k] + 1)
                if np.random.rand() <= mutation:
                    t_chromo2[i, j, k] = np.random.randint(0, t_chromo2[i, j, k] + 1)

    D = np.zeros((sources, destinations, modes), dtype=int)
    R = np.zeros((sources, destinations, modes), dtype=int)
    R1 = np.zeros((sources, destinations, modes), dtype=int)
    R2 = np.zeros((sources, destinations, modes), dtype=int)
    X1 = np.zeros((sources, destinations, modes), dtype=int)
    X2 = np.zeros((sources, destinations, modes), dtype=int)

    for i in range(sources):
        for j in range(destinations):
            for k in range(modes):
                D[i, j, k] = (t_chromo1[i, j, k] + t_chromo2[i, j, k]) // 2
                R[i, j, k] = (t_chromo1[i, j, k] + t_chromo2[i, j, k]) % 2
                if R[i, j, k] == 1:
                    count += 1
                    if count % 2 == 1:
                        R1[i, j, k] = R[i, j, k]
                    else:
                        R2[i, j, k] = R[i, j, k]

    for i in range(sources):
        for j in range(destinations):
            for k in range(modes):
                X1[i, j, k] = D[i, j, k] + R1[i, j, k]
                X2[i, j, k] = D[i, j, k] + R2[i, j, k]

    return X1, X2

def evaluate_fitness(chromosome, supply, demand, capacity):
    unused_supply = np.maximum(supply - np.sum(chromosome, axis=(1, 2)), 0).sum()
    unmet_demand = np.maximum(demand - np.sum(chromosome, axis=(0, 2)), 0).sum()
    unused_capacity = np.maximum(capacity - np.sum(chromosome, axis=(0, 1)), 0).sum()
    fitness_score = unused_supply + unmet_demand + unused_capacity
    return fitness_score

for i in range(population_size):
    chromosome = make_chromosome(pi, lenpi, supply, demand, capacity)
    population.append(chromosome)

print("\n\n# # # Initial Population # # #\n\n\n")
for idx, chromosome in enumerate(population):
    print(f"Chromosome {idx + 1}:\n\n{chromosome}\n\n\n")

for a in range(1, gen + 1):
    print(f"# # # Generation {a} Crossover/Mutation Population # # #\n\n")
    crossover = []
    crossover_count = 0

    for i in range(population_size):
        for j in range(i + 1, population_size):
            if crossover_count >= 10:
                break
            cross1, cross2 = make_crossover(population[i], population[j])
            crossover.append(cross1)
            crossover.append(cross2)
            crossover_count += 1
        if crossover_count >= 10:
            break

    for idx, cross in enumerate(crossover):
        print(f"Gen - {a}\nCrossover & Mutation {idx + 1}:\n\n{cross}\n\n\n")

    fitness_scores = []
    for chromo in crossover:
        fitness = evaluate_fitness(chromo, supply, demand, capacity)
        fitness_scores.append((chromo, fitness))
    sorted_chromosomes = sorted(fitness_scores, key=lambda x: x[1])
    top_10_solutions = sorted_chromosomes[:10]

    population = [chromo for chromo, fitness in top_10_solutions]

    print("\n\n# # # Top 10 Best Mutated Solutions # # #\n\n")
    for idx, (chromo, fitness) in enumerate(top_10_solutions, start=1):
        print(f"Gen - {a}\nRank {idx} - Fitness Score: {fitness}\n\n{chromo}\n\n\n")
