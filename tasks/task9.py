"""
Firefly Algorithm:
initialize population of fireflies = x_i (i=1,2,...,n)
set control parameters
t = 0
while t < t_max
    for i in range(n)
        for j in range(n) 
            light intensity I_i at x_i is determined by f(x_i)
            if I_j > I_i
                move firefly i towards j
            attractiveness varies with distance r via exp(-gamma r^2)
            evaluate new solutions and update light intensity
            
        rank fireflies and find the current best
        t += 1
"""
from solution import import_and_run
from tasks.solution import MAX_ITERATIONS
import numpy as np
import random

# Constants
POP_SIZE = 20  # population size
G = 200  # number of generations
ALPHA = 0.3  # alpha parameter for firefly movement
BETA_ZERO = 1.0  # minimum beta parameter for firefly movement
GAMMA = 1.0  # gamma parameter for attractiveness


def firefly_algorithm(lower_bound, upper_bound, function, max_iteration=MAX_ITERATIONS, d=2, pop_size=POP_SIZE, g_max=G,
                      alpha=ALPHA, b_0=BETA_ZERO, gamma=GAMMA):
    def generate_population():
        population = []
        for i in range(pop_size):
            population.append(np.array([random.uniform(lower_bound, upper_bound) for _ in range(d)]))
        return population

    def distance(a, b):
        return np.linalg.norm(a - b)

    def get_best(pop, lights):
        return pop[lights.index(min(lights))]

    def get_light_intensities(p):
        return [function(x) for x in p]

    def normal():
        return np.random.normal(0, 1, d)

    result = []
    pop = generate_population()
    lights = get_light_intensities(pop)

    for g in range(g_max):
        for i in range(pop_size):
            for j in range(pop_size):
                if lights[i] > lights[j]:  # If firefly i is brighter than firefly j
                    pop[i] = pop[i] + (b_0 / (1 + distance(pop[i], pop[j]))) * (pop[j] - pop[i]) + alpha * normal()

                lights[i] = function(pop[i])

        # Record the best solution for the current generation
        result.append([list(get_best(pop, lights))])

    return result


if __name__ == "__main__":
    import_and_run(firefly_algorithm)
