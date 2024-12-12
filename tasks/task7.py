import numpy as np
from task5 import generate_population
from solution import import_and_run

# Constants
POP_SIZE = 20
MIGRATIONS = 100
STEP = 0.11
PRT = 0.4
PATH_LENGTH = 3.0
DIMENSION = 2


def initialize_population(lower_bound, upper_bound, pop_size, dimension):
    """Generates the initial population within the given bounds."""
    return np.array(generate_population(lower_bound, upper_bound, pop_size, dimension))


def evaluate_population(population, test_function):
    """Finds the best individual and its value in the population."""
    best_individual = population[0]  # first individual is always the best
    best_value = test_function(best_individual)  # value of the best individual

    for individual in population:  # iterate through the population
        value = test_function(individual)  # calculate the value of the individual
        if value < best_value:  # if the value is better than the best value
            best_value = value  # update the best value
            best_individual = individual  # update the best individual

    return best_individual, best_value


def migrate_individual(individual, best_individual, test_function, lower_bound, upper_bound, step, path_length, prt):
    """Performs migration on a single individual."""
    for step_fraction in np.arange(0, path_length, step):  # iterate through the path of the individual
        prt_vector = np.random.rand(len(individual)) < prt  # generate a random vector of probabilities
        # calculate the new position of the individual based on the migration formula
        new_position = individual + step_fraction * (best_individual - individual) * prt_vector
        new_position = np.clip(new_position, lower_bound, upper_bound)  # clip the new position to the bounds

        if test_function(new_position) < test_function(individual):  # if the new position is better
            individual = new_position  # update the individual

    return individual


def self_organizing_migration_algorithm(lower_bound, upper_bound, test_function, pop_size=POP_SIZE, dim=DIMENSION,
                                        migrations=MIGRATIONS, step=STEP, prt=PRT, path_length=PATH_LENGTH):
    """SOMA all-to-one implementation."""
    population = initialize_population(lower_bound, upper_bound, pop_size, dim)  # generate the initial population
    best_individual, best_value = evaluate_population(population, test_function)  # find the best individual
    all_points = []

    for migration in range(migrations):  # iterate through the migrations
        new_population = []

        for individual in population:  # iterate through the population
            # perform migration on the individual, meaning update its position
            new_individual = migrate_individual(individual, best_individual, test_function, lower_bound,
                                                upper_bound, step, path_length, prt)
            new_population.append(new_individual)

        population = np.array(new_population)  # update the population
        all_points.append(new_population)  # save the population for visualization
        best_individual, best_value = evaluate_population(population, test_function)  # find the best individual

    return all_points


if __name__ == '__main__':
    import_and_run(self_organizing_migration_algorithm)
