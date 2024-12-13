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
    population = np.random.uniform(lower_bound, upper_bound, (pop_size, dimension))
    fitness = np.array([test_function(ind) for ind in population])
    ofe = pop_size
    results = []

    while ofe < max_ofe:
        best_idx = np.argmin(fitness)
        teacher = population[best_idx]
        mean_population = np.mean(population, axis=0)
        teaching_factor = np.random.randint(1, 3)

        new_population = population + np.random.rand(pop_size, dimension) * (
                teacher - teaching_factor * mean_population)
        new_population = np.clip(new_population, lower_bound, upper_bound)
        new_fitness = np.array([test_function(ind) for ind in new_population])
        ofe += pop_size

        improved = new_fitness < fitness
        population[improved] = new_population[improved]
        fitness[improved] = new_fitness[improved]

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

        results.append(population.copy().tolist())

    return results


def get_best_result(results, test_function):
    last_population = results[-1]
    fitness_values = [test_function(np.array(individual)) for individual in last_population]
    best_idx = np.argmin(fitness_values)
    return fitness_values[best_idx]


def save_results_to_excel(results, file_path):
    workbook = xlsxwriter.Workbook(file_path)

    for function_name, function_results in results.items():
        worksheet = workbook.add_worksheet(function_name)
        worksheet.write(0, 0, "Experiment")
        worksheet.write(0, 1, "DE")
        worksheet.write(0, 2, "PSO")
        worksheet.write(0, 3, "SOMA")
        worksheet.write(0, 4, "FA")
        worksheet.write(0, 5, "TLBO")

        for row_idx, row_data in enumerate(function_results, start=1):
            worksheet.write_row(row_idx, 0, row_data)

        data = np.array(function_results)[:, 1:]
        mean_row = ["Mean"] + np.mean(data, axis=0).tolist()
        std_row = ["Std. Dev"] + np.std(data, axis=0).tolist()
        worksheet.write_row(len(function_results) + 1, 0, mean_row)
        worksheet.write_row(len(function_results) + 2, 0, std_row)

    workbook.close()


def create_xlsx_file():
    results = {}

    for experiment_number in range(1, NUMBER_OF_EXPERIMENTS + 1):
        print(f"\nExperiment {experiment_number}/{NUMBER_OF_EXPERIMENTS}")

        for function_id, function in get_all_functions().items():
            lower_bound, upper_bound, _ = get_function_parameters(function)
            print(f"Function: {function.__name__}")

            differential_result = differential_evolution(lower_bound=lower_bound, upper_bound=upper_bound,
                                                         test_function=function, dimension=DIMENSION,
                                                         pop_size=POPULATION_SIZE, f=F, cr=CR, g_max=G_MAX)
            pso_result = particle_swarm_optimization(lower_bound=lower_bound, upper_bound=upper_bound,
                                                     test_function=function, pop_size=POPULATION_SIZE,
                                                     dimension=DIMENSION, m_max=M_MAX, c1=C1, c2=C2, v_max=V_MAX,
                                                     v_min=V_MIN, w_max=W_MAX, w_min=W_MIN)
            soma_result = self_organizing_migration_algorithm(lower_bound=lower_bound, upper_bound=upper_bound,
                                                              test_function=function, pop_size=POPULATION_SIZE,
                                                              dim=DIMENSION, migrations=MIGRATIONS, step=STEP, prt=PRT,
                                                              path_length=PATH_LENGTH)
            firefly_result = firefly_algorithm(lower_bound=lower_bound, upper_bound=upper_bound, test_function=function,
                                               d=DIMENSION, pop_size=POPULATION_SIZE, g_max=G, alpha=ALPHA,
                                               b_0=BETA_ZERO)
            tlbo_result = teaching_learning_based_optimization(lower_bound=lower_bound, upper_bound=upper_bound,
                                                               test_function=function, pop_size=POPULATION_SIZE,
                                                               dimension=DIMENSION, max_ofe=MAX_OFE)

            results.setdefault(function.__name__, []).append([
                experiment_number,
                get_best_result(differential_result, function),
                get_best_result(pso_result, function),
                get_best_result(soma_result, function),
                get_best_result(firefly_result, function),
                get_best_result(tlbo_result, function)
            ])

    save_dir = "../results/"
    os.makedirs(save_dir, exist_ok=True)
    save_results_to_excel(results, os.path.join(save_dir, "task10_results.xlsx"))


if __name__ == "__main__":
    create_xlsx_file()

    # uncomment this to create animations
    # import_and_run(teaching_learning_based_optimization)
