from solution import import_and_run

import copy
import random
import numpy as np

NP = 10  # number of individuals in the population
F = 0.5  # mutation constant
CR = 0.5  # crossover range
G_MAX = 100  # number of generation cycles


def generate_population(lower_bound, upper_bound, input_np, dimension=2):
    """Generate a population of NP individuals with the given dimension."""
    return [np.random.uniform(lower_bound, upper_bound, dimension).tolist() for _ in range(input_np)]


def get_random_parents(population, exclude):
    """Get random parents for mutation."""
    result = [i for i in range(len(population)) if population[i] not in exclude]
    return random.choice(result)


def differential_evolution(lower_bound, upper_bound, test_function, dimension=2, pop_size=NP, f=F, cr=CR, g_max=G_MAX):
    """Differential evolution algorithm for optimization in N dimensions."""
    result = []  # stores the population state at each generation
    pop = generate_population(lower_bound, upper_bound, pop_size, dimension)  # initial population
    for g in range(g_max):
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

            mutated_individual = (r1 - r2) * f + r3  # mutation formula
            trial_vector = np.zeros(len(individual))  # trial vector/individual initialized to zeros
            j_rnd = np.random.randint(0, len(individual))  # random index for crossover, ensures at least one crossover

            # crossover: generate trial vector by crossing over genes from mutated and original individual
            for j in range(len(individual)):
                # perform crossover based on crossover range and random index
                if np.random.uniform() < cr or j == j_rnd:
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
