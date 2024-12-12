from function import get_all_functions, get_function_parameters
from solution import import_and_run
from task5 import differential_evolution, generate_population
from task6 import particle_swarm_optimization
from task7 import self_organizing_migration_algorithm
from task9 import firefly_algorithm

import numpy as np

DIMENSION = 2
POPULATION_SIZE = 30
MAX_OFE = 3_000


def teaching_learning_based_optimization(lower_bound, upper_bound, test_function, pop_size=POPULATION_SIZE,
                                         dimension=DIMENSION, max_ofe=MAX_OFE):
    # Initialize population and fitness
    population = np.random.uniform(lower_bound, upper_bound, (pop_size, dimension))
    fitness = np.array([test_function(ind) for ind in population])
    ofe = pop_size  # Initial evaluations

    results = []  # Store population points for each iteration

    # Main optimization loop
    while ofe < max_ofe:
        # Teaching Phase
        best_idx = np.argmin(fitness)
        teacher = population[best_idx]

        mean_population = np.mean(population, axis=0)
        teaching_factor = np.random.randint(1, 3)

        new_population = population + np.random.rand(pop_size, dimension) * (
                teacher - teaching_factor * mean_population
        )
        new_population = np.clip(new_population, lower_bound, upper_bound)

        new_fitness = np.array([test_function(ind) for ind in new_population])
        ofe += pop_size

        # Accept better solutions
        improved = new_fitness < fitness
        population[improved] = new_population[improved]
        fitness[improved] = new_fitness[improved]

        # Learning Phase
        for i in range(pop_size):
            partner_idx = np.random.choice([idx for idx in range(pop_size) if idx != i])
            partner = population[partner_idx]

            direction = partner - population[i] if fitness[partner_idx] < fitness[i] else population[i] - partner

            new_individual = population[i] + np.random.rand(dimension) * direction
            new_individual = np.clip(new_individual, lower_bound, upper_bound)

            new_individual_fitness = test_function(new_individual)
            ofe += 1

            if new_individual_fitness < fitness[i]:
                population[i] = new_individual
                fitness[i] = new_individual_fitness

        # Store population points for this iteration
        results.append(population.copy().tolist())

    return results


if __name__ == "__main__":
    import_and_run(teaching_learning_based_optimization)
