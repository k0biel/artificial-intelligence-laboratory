from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *

def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


items, knapsack_max_capacity = get_big()
print(items)

population_size = 95
generations = 200
n_selection = 20
n_elite = 5

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)

for _ in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm
    new_population = []
    for _ in range((population_size - n_elite) // 2):
        # Choosing parents (roulette wheel selection)
        sum_fitness = 0
        fitness_table = []
        probability = []

        for individual in population:
            individual_fitness = fitness(items, knapsack_max_capacity, individual)
            fitness_table.append(individual_fitness)
            sum_fitness += individual_fitness

        for i in range(len(population)):
            probability.append(fitness_table[i] / sum_fitness)

        parent1, parent2 = random.choices(population, weights=probability, k=2)

        # Single-point crossing
        child1 = parent1[:len(parent1) // 2] + parent2[len(parent2) // 2:]
        child2 = parent2[:len(parent2) // 2] + parent1[len(parent1) // 2:]

        # Mutation
        mutation_point = random.randint(0, len(child1) - 1)
        child1[mutation_point] = not child1[mutation_point]
        mutation_point = random.randint(0, len(child2) - 1)
        child2[mutation_point] = not child2[mutation_point]

        new_population.append(child1)
        new_population.append(child2)

    # Update the population of solutions (adding an elite)
    population.sort(key=lambda individual: fitness(items, knapsack_max_capacity, individual), reverse=True)
    new_population.extend(population[:n_elite])

    population = new_population


    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
