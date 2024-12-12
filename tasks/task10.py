from function import get_all_functions, get_function_parameters
from solution import import_and_run
from task5 import differential_evolution, F, CR, G_MAX
from task6 import particle_swarm_optimization, M_MAX, C1, C2, V_MAX, V_MIN, W_MAX, W_MIN
from task7 import self_organizing_migration_algorithm, MIGRATIONS, STEP, PRT, PATH_LENGTH
from task9 import firefly_algorithm, G, ALPHA, BETA_ZERO

import numpy as np
import xlsxwriter
import os

DIMENSION = 30
POPULATION_SIZE = 30
MAX_OFE = 3_000
NUMBER_OF_EXPERIMENTS = 30


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


def get_best_result(results, test_function):
    last_population = results[-1]  # Get the last population
    fitness_values = [test_function(np.array(individual)) for individual in
                      last_population]  # Ensure it's a NumPy array
    best_idx = np.argmin(fitness_values)  # Find the index of the best individual
    best_value = fitness_values[best_idx]  # Get the best fitness value
    return best_value


def create_xlsx_file():
    results = []

    for experiment_number in range(1, NUMBER_OF_EXPERIMENTS + 1):
        print(f"Experiment {experiment_number}/{NUMBER_OF_EXPERIMENTS}")
        for i, function in get_all_functions().items():
            lower_bound, upper_bound, _ = get_function_parameters(function)
            print(f"Function: {function.__name__}")

            differential_result = differential_evolution(lower_bound=lower_bound, upper_bound=upper_bound,
                                                         test_function=function, dimension=DIMENSION,
                                                         pop_size=POPULATION_SIZE, f=F, cr=CR, g_max=G_MAX)
            dif_best = get_best_result(differential_result, function)

            pso_res = particle_swarm_optimization(lower_bound=lower_bound, upper_bound=upper_bound,
                                                  test_function=function,
                                                  pop_size=POPULATION_SIZE, dimension=DIMENSION, m_max=M_MAX, c1=C1,
                                                  c2=C2, v_max=V_MAX, v_min=V_MIN, w_max=W_MAX, w_min=W_MIN)
            pso_best = get_best_result(pso_res, function)

            soma_res = self_organizing_migration_algorithm(lower_bound=lower_bound, upper_bound=upper_bound,
                                                           test_function=function, pop_size=POPULATION_SIZE,
                                                           dim=DIMENSION, migrations=MIGRATIONS, step=STEP, prt=PRT,
                                                           path_length=PATH_LENGTH)
            soma_best = get_best_result(soma_res, function)

            firefly_res = firefly_algorithm(lower_bound=lower_bound, upper_bound=upper_bound, test_function=function,
                                            d=DIMENSION, pop_size=POPULATION_SIZE, g_max=G, alpha=ALPHA, b_0=BETA_ZERO)
            firefly_best = get_best_result(firefly_res, function)

            tlbo_res = teaching_learning_based_optimization(lower_bound=lower_bound,
                                                            upper_bound=upper_bound,
                                                            test_function=function,
                                                            pop_size=POPULATION_SIZE,
                                                            dimension=DIMENSION, max_ofe=MAX_OFE)
            tlbo_best = get_best_result(tlbo_res, function)

            results.append((experiment_number, function.__name__,
                            dif_best, pso_best, soma_best, firefly_best, tlbo_best))

        print()

    # create xlsx file
    save_dir = "../results/"
    file_name = "task10_results.xlsx"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    workbook = xlsxwriter.Workbook(save_dir + file_name)

    # create worksheet for each function
    for function in get_all_functions().values():
        worksheet = workbook.add_worksheet(function.__name__)
        worksheet.write(0, 0, "Experiment")
        worksheet.write(0, 1, "DE")
        worksheet.write(0, 2, "PSO")
        worksheet.write(0, 3, "SOMA")
        worksheet.write(0, 4, "FA")
        worksheet.write(0, 5, "TLBO")

    # write results to the xlsx file
    for result in results:
        worksheet = workbook.get_worksheet_by_name(result[1])
        worksheet.write(result[0], 0, f"Exp. {result[0]}")
        worksheet.write(result[0], 1, result[2])  # Just write the scalar directly
        worksheet.write(result[0], 2, result[3])  # Same here
        worksheet.write(result[0], 3, result[4])  # And here
        worksheet.write(result[0], 4, result[5])  # Same here
        worksheet.write(result[0], 5, result[6])  # Same here

    # Calculate mean and std deviation and put it in the last row
    for function in get_all_functions().values():
        worksheet = workbook.get_worksheet_by_name(function.__name__)
        worksheet.write(NUMBER_OF_EXPERIMENTS + 1, 0, "Mean")
        worksheet.write(NUMBER_OF_EXPERIMENTS + 2, 0, "Std. Dev")

        # Collect data manually to calculate mean and std. dev
        data_de = [result[2] for result in results if result[1] == function.__name__]
        data_pso = [result[3] for result in results if result[1] == function.__name__]
        data_soma = [result[4] for result in results if result[1] == function.__name__]
        data_fa = [result[5] for result in results if result[1] == function.__name__]
        data_tlbo = [result[6] for result in results if result[1] == function.__name__]

        # Write the mean and standard deviation for each column
        for col, data in enumerate([data_de, data_pso, data_soma, data_fa, data_tlbo], 1):
            mean = np.mean(data)
            std_dev = np.std(data)
            worksheet.write(NUMBER_OF_EXPERIMENTS + 1, col, mean)
            worksheet.write(NUMBER_OF_EXPERIMENTS + 2, col, std_dev)

    workbook.close()


if __name__ == "__main__":
    # uncomment to create and save animations, maybe check and set dimension value to 2
    # import_and_run(teaching_learning_based_optimization)

    # uncomment to create xlsx file with all the results
    create_xlsx_file()
