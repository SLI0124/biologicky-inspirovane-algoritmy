import numpy as np
import random
from solution import import_and_run

# Constants
POP_SIZE = 20  # Population size
G = 200  # Number of generations
ALPHA = 0.3  # Alpha for the firefly movement, that is the randomization parameter
BETA_ZERO = 1.0  # Beta for the firefly movement, that is the attractiveness parameter

DIMENSION = 2  # Dimension of the search space


def generate_population(lower_bound, upper_bound, d, pop_size):
    """Generate an initial population of fireflies."""
    return [np.array([random.uniform(lower_bound, upper_bound) for _ in range(d)]) for _ in range(pop_size)]


def distance(x1, x2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(x1 - x2)


def get_light_intensities(pop, function):
    """Calculate the light intensities of the fireflies."""
    return [function(x) for x in pop]


def normal(d):
    """Generate a normally distributed random vector."""
    return np.random.normal(0, 1, d)


def firefly_algorithm(lower_bound, upper_bound, test_function, d=DIMENSION, pop_size=POP_SIZE,
                      g_max=G, alpha=ALPHA, b_0=BETA_ZERO):
    pop = generate_population(lower_bound, upper_bound, d, pop_size)  # Generate the initial population
    lights = get_light_intensities(pop, test_function)  # Calculate the light intensities of the fireflies
    result = []  # Store the positions of all fireflies for each generation

    for _ in range(g_max):  # Iterate over the generations
        generation_points = []  # Store the positions of fireflies for the current generation
        for i in range(pop_size):  # Iterate over the fireflies
            for j in range(pop_size):  # Iterate over the other fireflies
                if lights[i] > lights[j]:  # If the light intensity of the current firefly is greater than the other
                    # Update the position of the firefly
                    pop[i] += (b_0 / (1 + distance(pop[i], pop[j]))) * (pop[j] - pop[i]) + alpha * normal(d)
            lights[i] = test_function(pop[i])  # Update the light intensity of the firefly
            generation_points.append(list(pop[i]))  # Store the position of the firefly
        result.append(generation_points)  # Store the positions of all fireflies for the current generation

    return result


if __name__ == "__main__":
    import_and_run(firefly_algorithm)
