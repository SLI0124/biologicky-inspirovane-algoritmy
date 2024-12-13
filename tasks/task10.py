from function import get_all_functions, get_function_parameters
from solution import import_and_run
from task5 import differential_evolution, F, CR, G_MAX
from task6 import particle_swarm_optimization, M_MAX, C1, C2, V_MAX, V_MIN, W_MAX, W_MIN
from task7 import self_organizing_migration_algorithm, MIGRATIONS, STEP, PRT, PATH_LENGTH
from task9 import firefly_algorithm, G, ALPHA, BETA_ZERO

import numpy as np
import xlsxwriter
import os

DIMENSION = 2
POPULATION_SIZE = 30
MAX_OFE = 3_000
NUMBER_OF_EXPERIMENTS = 30


def teaching_learning_based_optimization(lower_bound, upper_bound, test_function, pop_size=POPULATION_SIZE,
                                         dimension=DIMENSION, max_ofe=MAX_OFE):
    """
    Implementace optimalizačního algoritmu Teaching-Learning-Based Optimization (TLBO).

    TLBO je metaheuristický algoritmus inspirovaný procesem výuky a učení. Algoritmus pracuje ve dvou hlavních fázích:
    1. Fáze učitele: Jedinec s nejlepší fitness hodnotou působí jako učitel, který se snaží přiblížit populaci k lepším hodnotám.
    2. Fáze studenta: Jedinci se navzájem ovlivňují a snaží se zlepšit na základě interakcí mezi sebou.

    Parametry:
        lower_bound (float): Dolní hranice hodnot proměnných.
        upper_bound (float): Horní hranice hodnot proměnných.
        test_function (callable): Testovací funkce, která hodnotí kvalitu jednotlivých řešení (fitness).
        pop_size (int): Počet jedinců v populaci (velikost populace).
        dimension (int): Počet dimenzí problému (počet proměnných).
        max_ofe (int): Maximální počet vyhodnocení fitness funkce (Objective Function Evaluations, OFE).

    Návratová hodnota:
        results (list): Historie populací během iterací. Každý prvek seznamu obsahuje kopii populace po dané iteraci.
    """

    # Inicializace populace s náhodnými hodnotami v rámci dolní a horní hranice.
    population = np.random.uniform(lower_bound, upper_bound, (pop_size, dimension))
    # Výpočet fitness hodnot pro všechny jedince v populaci.
    fitness = np.array([test_function(ind) for ind in population])
    # Počáteční počet vyhodnocení fitness funkce.
    ofe = pop_size
    # Seznam pro ukládání historických populací.
    results = []

    # Hlavní smyčka běží, dokud nedosáhneme maximálního počtu vyhodnocení fitness.
    while ofe < max_ofe:
        # Určení nejlepšího jedince v populaci (učitel).
        best_idx = np.argmin(fitness)
        teacher = population[best_idx]
        # Výpočet průměrné hodnoty populace v každé dimenzi.
        mean_population = np.mean(population, axis=0)
        # Náhodné určení učebního faktoru (1 nebo 2).
        teaching_factor = np.random.randint(1, 3)

        # Fáze učitele: Zlepšení jedinců směrem k učiteli.
        new_population = population + np.random.rand(pop_size, dimension) * (
                teacher - teaching_factor * mean_population)
        # Oříznutí nových hodnot do povoleného rozsahu (dolní a horní hranice).
        new_population = np.clip(new_population, lower_bound, upper_bound)
        # Výpočet fitness pro novou populaci.
        new_fitness = np.array([test_function(ind) for ind in new_population])
        ofe += pop_size

        # Aktualizace populace: Zlepšení jedinců s lepší fitness hodnotou.
        improved = new_fitness < fitness
        population[improved] = new_population[improved]
        fitness[improved] = new_fitness[improved]

        # Fáze studenta: Jedinci se snaží učit jeden od druhého.
        for i in range(pop_size):
            # Výběr náhodného partnera z populace (jiného než aktuální jedinec).
            partner_idx = np.random.choice([idx for idx in range(pop_size) if idx != i])
            partner = population[partner_idx]

            # Směr zlepšení závisí na fitness hodnotě partnera a aktuálního jedince.
            if fitness[partner_idx] < fitness[i]:
                direction = partner - population[i]
            else:
                direction = population[i] - partner

            # Výpočet nové hodnoty jedince na základě směru a náhodné změny.
            new_individual = population[i] + np.random.rand(dimension) * direction
            # Oříznutí nové hodnoty do povoleného rozsahu.
            new_individual = np.clip(new_individual, lower_bound, upper_bound)
            # Výpočet fitness nové hodnoty.
            new_individual_fitness = test_function(new_individual)
            ofe += 1

            # Aktualizace jedince, pokud má nová hodnota lepší fitness.
            if new_individual_fitness < fitness[i]:
                population[i] = new_individual
                fitness[i] = new_individual_fitness

        # Uložení aktuální populace do historie výsledků.
        results.append(population.copy().tolist())

    # Návrat historických populací.
    return results


def get_best_result(results, test_function):
    """
    Funkce pro nalezení nejlepší fitness hodnoty z poslední populace.

    Parametry:
        results (list): Seznam populací během iterací. Každá populace je reprezentována jako seznam jedinců (řešení).
                        Poslední prvek seznamu `results` obsahuje populaci po poslední iteraci algoritmu.
        test_function (callable): Testovací funkce, která vyhodnocuje kvalitu každého jedince (fitness).

    Návratová hodnota:
        float: Nejlepší fitness hodnota z poslední populace.
    """

    # Extrahuje poslední populaci z výsledků (populace po poslední iteraci).
    last_population = results[-1]

    # Vypočítá fitness hodnoty pro všechny jedince v poslední populaci.
    fitness_values = [test_function(np.array(individual)) for individual in last_population]

    # Najde index nejlepší fitness hodnoty (nejmenší hodnota u minimalizačních problémů).
    best_idx = np.argmin(fitness_values)

    # Vrátí nejlepší fitness hodnotu.
    return fitness_values[best_idx]


def save_results_to_excel(results, file_path):
    """
    Uloží výsledky experimentů do Excelového souboru, včetně průměru a směrodatné odchylky výsledků.

    Parametry:
        results (dict): Slovník, kde klíče jsou názvy testovacích funkcí a hodnoty jsou seznamy výsledků experimentů.
                        Každý výsledek experimentu je seznam, který obsahuje:
                        - číslo experimentu,
                        - nejlepší hodnoty fitness z různých algoritmů (např. DE, PSO, SOMA, FA, TLBO).
        file_path (str): Cesta k výslednému Excel souboru, kam se výsledky uloží.

    Návratová hodnota:
        None: Funkce nevykazuje žádnou návratovou hodnotu. Výsledky jsou uloženy do souboru.
    """

    # Vytvoření nového Excelového souboru pomocí knihovny xlsxwriter.
    workbook = xlsxwriter.Workbook(file_path)

    # Iterace přes všechny testovací funkce ve slovníku výsledků.
    for function_name, function_results in results.items():
        # Vytvoření nového listu v Excelu pro každou testovací funkci.
        worksheet = workbook.add_worksheet(function_name)

        # Zápis hlaviček sloupců (názvy algoritmů).
        worksheet.write(0, 0, "Experiment")  # Sloupec s čísly experimentů.
        worksheet.write(0, 1, "DE")  # Diferenciální evoluce.
        worksheet.write(0, 2, "PSO")  # Particle Swarm Optimization.
        worksheet.write(0, 3, "SOMA")  # Self-Organizing Migration Algorithm.
        worksheet.write(0, 4, "FA")  # Firefly Algorithm.
        worksheet.write(0, 5, "TLBO")  # Teaching-Learning-Based Optimization.

        # Zápis výsledků experimentů do Excelového listu.
        for row_idx, row_data in enumerate(function_results, start=1):
            # Každý řádek obsahuje číslo experimentu a výsledky algoritmů.
            worksheet.write_row(row_idx, 0, row_data)

        # Výpočet statistických hodnot (průměr a směrodatná odchylka) pro jednotlivé algoritmy.
        # Data jsou extrahována (bez prvního sloupce, který obsahuje čísla experimentů).
        data = np.array(function_results)[:, 1:]  # Extrahuje pouze výsledky algoritmů.

        # Vytvoření řádku s průměrnými hodnotami.
        mean_row = ["Mean"] + np.mean(data, axis=0).tolist()
        # Vytvoření řádku se směrodatnými odchylkami.
        std_row = ["Std. Dev"] + np.std(data, axis=0).tolist()

        # Zápis řádku s průměrem na konec tabulky.
        worksheet.write_row(len(function_results) + 1, 0, mean_row)
        # Zápis řádku se směrodatnou odchylkou pod průměr.
        worksheet.write_row(len(function_results) + 2, 0, std_row)

    # Uzavření a uložení Excelového souboru.
    workbook.close()


def create_xlsx_file():
    """
    Vytvoří Excelový soubor obsahující výsledky porovnání optimalizačních algoritmů.

    Funkce provádí následující kroky:
        1. Provede několik experimentů (daný počet `NUMBER_OF_EXPERIMENTS`).
        2. Pro každý experiment a každou testovací funkci spustí různé optimalizační algoritmy.
        3. Výsledky experimentů uloží do Excelového souboru s přehledem nejlepší fitness hodnot z jednotlivých algoritmů.

    Parametry:
        Žádné (funkce používá globální konstanty pro konfiguraci).

    Návratová hodnota:
        None: Výsledky jsou uloženy do souboru, funkce nevykazuje žádnou návratovou hodnotu.
    """

    # Slovník pro uložení výsledků všech experimentů a funkcí.
    results = {}

    # Iterace přes zadaný počet experimentů.
    for experiment_number in range(1, NUMBER_OF_EXPERIMENTS + 1):
        print(f"\nExperiment {experiment_number}/{NUMBER_OF_EXPERIMENTS}")

        # Iterace přes všechny dostupné testovací funkce.
        for function_id, function in get_all_functions().items():
            # Získání parametrů testovací funkce (dolní a horní hranice a rozměr prostoru).
            lower_bound, upper_bound, _ = get_function_parameters(function)
            print(f"Function: {function.__name__}")

            # Spuštění jednotlivých optimalizačních algoritmů:
            # 1. Diferenciální evoluce.
            differential_result = differential_evolution(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                test_function=function,
                dimension=DIMENSION,
                pop_size=POPULATION_SIZE,
                f=F,
                cr=CR,
                g_max=G_MAX
            )

            # 2. Particle Swarm Optimization.
            pso_result = particle_swarm_optimization(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                test_function=function,
                pop_size=POPULATION_SIZE,
                dimension=DIMENSION,
                m_max=M_MAX,
                c1=C1,
                c2=C2,
                v_max=V_MAX,
                v_min=V_MIN,
                w_max=W_MAX,
                w_min=W_MIN
            )

            # 3. Self-Organizing Migration Algorithm (SOMA).
            soma_result = self_organizing_migration_algorithm(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                test_function=function,
                pop_size=POPULATION_SIZE,
                dim=DIMENSION,
                migrations=MIGRATIONS,
                step=STEP,
                prt=PRT,
                path_length=PATH_LENGTH
            )

            # 4. Firefly Algorithm.
            firefly_result = firefly_algorithm(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                test_function=function,
                d=DIMENSION,
                pop_size=POPULATION_SIZE,
                g_max=G,
                alpha=ALPHA,
                b_0=BETA_ZERO
            )

            # 5. Teaching-Learning-Based Optimization.
            tlbo_result = teaching_learning_based_optimization(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                test_function=function,
                pop_size=POPULATION_SIZE,
                dimension=DIMENSION,
                max_ofe=MAX_OFE
            )

            # Uložení výsledků experimentu pro aktuální testovací funkci.
            # Každý řádek obsahuje číslo experimentu a nejlepší hodnoty fitness z jednotlivých algoritmů.
            results.setdefault(function.__name__, []).append([
                experiment_number,
                get_best_result(differential_result, function),
                get_best_result(pso_result, function),
                get_best_result(soma_result, function),
                get_best_result(firefly_result, function),
                get_best_result(tlbo_result, function)
            ])

    # Uložení výsledků do Excelového souboru:
    # Výsledkový adresář je vytvořen, pokud neexistuje.
    save_dir = "../results/"
    os.makedirs(save_dir, exist_ok=True)

    # Výsledky jsou uloženy do souboru "task10_results.xlsx".
    save_results_to_excel(results, os.path.join(save_dir, "task10_results.xlsx"))


if __name__ == "__main__":
    # un/comment this to create animations
    import_and_run(teaching_learning_based_optimization)

    # un/comment this to create xlsx file (it will take some time)
    # and change the desired number of experiments, dimensions or population size
    # create_xlsx_file()
