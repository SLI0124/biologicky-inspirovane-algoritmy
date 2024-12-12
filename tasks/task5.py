from solution import import_and_run

import copy
import random
import numpy as np

NP = 10  # number of individuals in the population
F = 0.5  # mutation constant
CR = 0.5  # crossover range
G_MAX = 100  # number of generation cycles


def generate_population(lower_bound, upper_bound, input_np):
    """Generate a population of NP individuals."""
    result = []
    for _ in range(input_np):
        x = lower_bound + (upper_bound - lower_bound) * np.random.rand()
        y = lower_bound + (upper_bound - lower_bound) * np.random.rand()
        result.append([x, y])
    return result


def get_random_parents(population, exclude):
    """Get random parents for mutation."""
    result = []
    for i in range(len(population)):
        if population[i] not in exclude:
            result.append(i)

    return random.choice(result)


def differential_evolution(lower_bound, upper_bound, test_function):
    """Differential evolution algorithm for optimization."""
    result = []  # stores the population state at each generation
    pop = generate_population(lower_bound, upper_bound, NP)  # initial population
    for g in range(G_MAX):
        new_population = copy.deepcopy(pop)

        # iterate over all individuals in the population
        for i, individual in enumerate(pop):
            # mutation: select three unique random individuals from the population for mutation
            r1_i = get_random_parents(pop, [individual])  # first random parent index
            r2_i = get_random_parents(pop, [individual, pop[r1_i]])
            r3_i = get_random_parents(pop, [individual, pop[r1_i], pop[r2_i]])

            # convert to numpy arrays for better vector operations
            r1 = np.array(new_population[r1_i])
            r2 = np.array(new_population[r2_i])
            r3 = np.array(new_population[r3_i])

            mutated_individual = (r1 - r2) * F + r3  # mutation formula
            trial_vector = np.zeros(len(individual))  # trial vector/individual initialized to zeros
            j_rnd = np.random.randint(0, len(individual))  # random index for crossover, ensures at least one crossover

            # crossover: generate trial vector by crossing over genes from mutated and original individual
            for j in range(len(individual)):
                # preform crossover based on crossover range and random index
                if np.random.uniform() < CR or j == j_rnd:
                    trial_vector[j] = mutated_individual[j]
                else:
                    trial_vector[j] = individual[j]

            trial_vector = np.clip(trial_vector, lower_bound, upper_bound)

            # selection: replace individual with trial vector if it has a better fitness
            if test_function(np.array(trial_vector)) <= test_function(np.array(individual)):
                new_population[i] = list(trial_vector)

        result.append(copy.deepcopy(new_population))  # store the population state
        pop = new_population  # move the new population to the current population

    return result  # evaluation history of the population


if __name__ == '__main__':
    import_and_run(differential_evolution)
