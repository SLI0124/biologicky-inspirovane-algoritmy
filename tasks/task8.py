import copy
import random
import numpy as np

from task4 import X_RANGE, Y_RANGE, generate_random_cities, G, D, calculate_distance, evaluate_individual, animate

# Constants
# D - number of ants
# G - number of generations
P = 0.5  # Evaporation rate


def distance_matrix(cities):
    """Generate a distance matrix for the cities"""
    matrix = np.zeros((len(cities), len(cities)))  # initialize the matrix with zeros
    for i in range(len(cities)):  # iterate over the cities
        for j in range(len(cities)):  # iterate over the cities
            if i != j:  # different cities than itself
                distance = calculate_distance(cities[i], cities[j])  # calculate the distance between the cities
                matrix[i][j] = distance  # set the distance to the matrix
                matrix[j][i] = distance  # set the distance to the matrix on the other side as well
            else:
                matrix[i][j] = 0  # set the distance to 0 since it's the same city

    return matrix


def get_best_path(cities, ants):
    """Get the best path from the ants"""

    def transform(x):
        """Transform the indices to cities"""
        return [cities[y] for y in x]

    best_path = transform(ants[0])  # initialize the best path with the first ant
    for ant in ants:  # iterate over the ants
        transformed = transform(ant)  # transform the indices to cities
        if evaluate_individual(transformed) < evaluate_individual(best_path):  # if the path is better
            best_path = transformed  # update the best path

    return best_path


def main():
    results = []  # initialize the results list to store the best paths
    cities = generate_random_cities(D, X_RANGE, Y_RANGE)  # initialize the first generation of cities
    pheromone_matrix = np.ones((D, D))  # initialize pheromone matrix that stores the pheromones, that guides the ants

    for i in range(G):  # iterate over the number of generations
        matrix = distance_matrix(cities)  # generate distance matrix
        # calculate the inverse of the distance matrix
        inverse_distances = np.divide(pheromone_matrix, matrix,
                                      out=np.zeros_like(pheromone_matrix), where=matrix != 0)
        ants = []  # initialize the ants list to store the paths

        for j in range(D):  # iterate over the number of ants
            visibility_matrix = copy.deepcopy(inverse_distances)  # copy the inverse distance matrix
            visibility_matrix[:, j] = 0  # set the visibility of the current city to 0
            visited_cities = [j]  # initialize the visited cities with the current city

            for k in range(D - 1):  # iterate over the number of cities
                summary = 0  # initialize the summary to 0 to calculate the probabilities
                options = []  # initialize the options list to store the possible cities
                for l in range(len(visibility_matrix[visited_cities[-1]])):  # iterate over the visibility matrix
                    val = visibility_matrix[visited_cities[-1]][l]  # get the visibility value
                    if val > 0:  # if the visibility value is greater than 0
                        options.append(l)  # add the city to the options list
                        summary += pheromone_matrix[visited_cities[-1]][l] * (val ** 2)  # calculate the summary

                probabilities = []  # initialize the probabilities list to store the probabilities for the cities
                for l in range(len(visibility_matrix[visited_cities[-1]])):  # iterate over the visibility matrix
                    val = visibility_matrix[visited_cities[-1]][l]  # get the visibility value
                    if val > 0:  # if the visibility value is greater than 0
                        probabilities.append(val ** 2 / summary)  # calculate the probability and add it to the list

                next_city = random.choices(options, probabilities)  # select the next city based on the probabilities
                visibility_matrix[:, next_city[0]] = 0  # set the visibility of the next city to 0
                visited_cities.append(next_city[0])  # add the next city to the visited cities

            visited_cities.append(visited_cities[0])  # add the first city to the end of the visited cities
            ants.append(visited_cities)  # add the visited cities to the ants list

        results.append(get_best_path(cities, ants))  # append the best path to the results list

        pheromone_matrix = pheromone_matrix * (1 - P)  # evaporate the pheromones
        for j in range(D):  # iterate over the number of ants
            distance = evaluate_individual([cities[x] for x in ants[j]])  # calculate the distance of the path
            for k in range(len(ants[j]) - 1):  # iterate over the cities
                pheromone_matrix[ants[j][k]][ants[j][k + 1]] += 1 / distance  # update the pheromone matrix

        print(f"Generation {i + 1}/{G}, Best Path Distance: {evaluate_individual(results[-1]):.2f}")

    animate(results, cities, "Ant Colony Optimization", "../animations/ant_colony_tsp/")


if __name__ == "__main__":
    main()
