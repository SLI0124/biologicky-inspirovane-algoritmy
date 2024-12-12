from function import get_all_functions, get_function_parameters
from solution import import_and_run
from task5 import differential_evolution, F, CR, G_MAX
from task6 import particle_swarm_optimization, M_MAX, C1, C2, V_MAX, V_MIN, W_MAX, W_MIN
from task7 import self_organizing_migration_algorithm, MIGRATIONS, STEP, PRT, PATH_LENGTH
from task9 import firefly_algorithm, G, ALPHA, BETA_ZERO

import numpy as np
import time
import xlsxwriter
import os

DIMENSION = 30
POPULATION_SIZE = 30
MAX_OFE = 3_000
NUMBER_OF_EXPERIMENTS = 2


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


def create_xlsx_file():
    results = []

    for exp_count in range(1, NUMBER_OF_EXPERIMENTS + 1):
        print(f"Experiment {exp_count}/{NUMBER_OF_EXPERIMENTS}")
        for i, function in get_all_functions().items():
            lower_bound, upper_bound, _ = get_function_parameters(function)
            print(f"Function: {function.__name__}")

            differential_result = differential_evolution(lower_bound=lower_bound, upper_bound=upper_bound,
                                                         test_function=function, dimension=DIMENSION,
                                                         pop_size=POPULATION_SIZE)

            pso_res = particle_swarm_optimization(lower_bound=lower_bound, upper_bound=upper_bound,
                                                  test_function=function,
                                                  pop_size=POPULATION_SIZE, dimension=DIMENSION)

            soma_res = self_organizing_migration_algorithm(lower_bound=lower_bound, upper_bound=upper_bound,
                                                           test_function=function, pop_size=POPULATION_SIZE,
                                                           dim=DIMENSION)

            firefly_res = firefly_algorithm(lower_bound=lower_bound, upper_bound=upper_bound, test_function=function,
                                            d=DIMENSION, pop_size=POPULATION_SIZE)

            teach_learn_based_res = teaching_learning_based_optimization(lower_bound=lower_bound,
                                                                         upper_bound=upper_bound,
                                                                         test_function=function,
                                                                         pop_size=POPULATION_SIZE,
                                                                         dimension=DIMENSION, max_ofe=MAX_OFE)

            # pick the best solution for each algorithm, we assume that algorithms found best solution as last element
            # of the results, second element is result value, for example (x, y, result) we want to always pick the last
            # element, no matter the dimension of the result
            best_diff = differential_result[-1][-1]
            best_pso = pso_res[-1][-1]
            best_soma = soma_res[-1][-1]
            best_firefly = firefly_res[-1][-1]
            best_teach_learn_based = teach_learn_based_res[-1][-1]

            # append results
            results.append(
                [exp_count, function.__name__, best_diff, best_pso, best_soma, best_firefly, best_teach_learn_based])

        print()

    # print results
    for result in results:
        print(f"Experiment {result[0]}\tFunction: {result[1]}\nBest DE: {result[2]}\nBest PSO: {result[3]}\n"
              f"Best SOMA: {result[4]}\nBest FA: {result[5]}\nBest TLBO: {result[6]}\n")

    save_path = "../results/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = "task10_results.xlsx"

    workbook = xlsxwriter.Workbook(save_path + file_name)

    # create 9 worksheets for each function
    # start by adding title to the worksheet that is 6 columns wide

    # then create header for each worksheet with 6 columns:
    # empty (for experiment columns), DE, PSO, SOMA, FA TLBO

    # put results in the worksheet accordingly

    # at the end, calculate mean and std. deviation for each algorithm and each function
    # put the name instead of experiment number

    for i, function in get_all_functions().items():
        worksheet = workbook.add_worksheet(function.__name__)

        # create title 6 columns wide
        worksheet.merge_range(first_row=0, first_col=0, last_row=0, last_col=5, data=function.__name__,
                              cell_format=workbook.add_format({'bold': True, 'font_size': 16}))

        # worksheet.write(1, 0, "")
        worksheet.write(row=1, col=0, data="Experiment")
        worksheet.write(row=1, col=1, data="DE")
        worksheet.write(row=1, col=2, data="PSO")
        worksheet.write(row=1, col=3, data="SOMA")
        worksheet.write(row=1, col=4, data="FA")
        worksheet.write(row=1, col=5, data="TLBO")

        for j, result in enumerate(results):  # for each result
            if result[1] == function.__name__:
                worksheet.write(row=j + 2, col=0, data="Experiment " + str(result[0]))
                worksheet.write(row=j + 2, col=1, data=result[2])
                worksheet.write(row=j + 2, col=2, data=result[3])
                worksheet.write(row=j + 2, col=3, data=result[4])
                worksheet.write(row=j + 2, col=4, data=result[5])
                worksheet.write(row=j + 2, col=5, data=result[6])

    # calculate mean and std. deviation for each algorithm and each function
    # for i, function in get_all_functions().items():
    #     worksheet = workbook.get_worksheet_by_name(function.__name__)
    #     worksheet.write(len(results) + 1, 0, "Mean")
    #     worksheet.write(len(results) + 2, 0, "Std. Deviation")
    #
    #     for j in range(1, 6):
    #         values = [result[j] for result in results if result[1] == function.__name__]
    #         values = [v for v in values if isinstance(v, (int, float))]
    #         worksheet.write(len(results) + 1, j, np.mean(values))
    #         worksheet.write(len(results) + 2, j, np.std(values))

    workbook.close()


if __name__ == "__main__":
    # uncomment to create and save animations, maybe check and set dimension value to 2
    # import_and_run(teaching_learning_based_optimization)

    # uncomment to create xlsx file with all the results
    create_xlsx_file()
