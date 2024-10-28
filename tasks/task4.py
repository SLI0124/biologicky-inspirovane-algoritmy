import os
import random
import math
import copy

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
NP = 20
G = 200
D = 20
X_RANGE = 0, 100
Y_RANGE = 0, 100


def calculate_distance(city1, city2):
    """Calculate the Euclidean distance between two cities"""
    x1, y1 = city1
    x2, y2 = city2
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


def generate_random_cities(n, x_range=X_RANGE, y_range=Y_RANGE):
    """Generate random cities within the given range"""
    cities = list()
    for i in range(n):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        cities.append((x, y))
    return cities


def generate_random_individual(cities):
    """Generate a random individual by shuffling the cities"""
    individual = copy.deepcopy(cities)
    random.shuffle(individual)
    return individual


def generate_random_population(cities, n=NP):
    """Generate a random population of individuals, this basically calls generate_random_individual n times"""
    population = list()
    for i in range(n):
        individual = generate_random_individual(cities)
        population.append(individual)
    return population


def order_crossover(parent_a, parent_b):
    """Perform Order Crossover  on two parents to generate offspring"""
    size = len(parent_a)  # size of the parent for the offspring
    start, end = sorted(random.sample(range(size), 2))  # select two random indices to slice the parent

    offspring = [None] * size  # initialize empty offspring with None values of the size of the parent
    offspring[start:end] = parent_a[start:end]  # copy the sliced part of parent to the offspring

    fill_pos = end  # start filling the offspring from the end of the sliced part
    for city in parent_b:  # iterate over the parent b since we have already copied the sliced part of parent a
        if city not in offspring:  # if the city is not already in the offspring, prevent duplicates
            if fill_pos >= size:  # if the fill position is greater than the size of the offspring
                fill_pos = 0  # reset the fill position to 0
            offspring[fill_pos] = city  # fill the offspring with the city from parent b
            fill_pos += 1  # increment the fill position to fill the next city

    return offspring


def mutate(offspring):
    """Mutate the offspring by swapping two random cities"""
    i, j = random.sample(range(len(offspring)), 2)
    offspring[i], offspring[j] = offspring[j], offspring[i]
    return offspring


def evaluate_individual(individual):
    """Evaluate the individual by calculating the total distance of the path"""
    total_distance = 0
    for i in range(len(individual) - 1):
        total_distance += calculate_distance(individual[i], individual[i + 1])
    return total_distance


def get_best_generation(population):
    """Get the best individual in the population"""
    best_individual = population[0]
    for individual in population:
        if evaluate_individual(individual) < evaluate_individual(best_individual):
            best_individual = individual
    return best_individual


def animate(results, cities):
    """Plot the results in form of animation and save it as gif"""
    fig, ax = plt.subplots()
    x, y = zip(*results[0])
    line, = ax.plot(x, y, 'b-', marker='o', markerfacecolor='red', markersize=5)
    ax.set_xlim(X_RANGE)
    ax.set_ylim(Y_RANGE)
    ax.set_title('Traveling Salesman Problem using Genetic Algorithm')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    city_x, city_y = zip(*cities)  # Plot the cities as fixed points
    ax.plot(city_x, city_y, 'go', markersize=8)

    # Position the text in a non-overlapping location
    text = ax.text(0.05, 0.85, '', transform=ax.transAxes,
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.5"))

    # helper function to update the plot for each frame
    def update(frame):
        if frame < len(results):
            x_update, y_update = zip(*results[frame])
            line.set_data(x_update, y_update)
            distance = evaluate_individual(results[frame])

            text.set_text(  # Set structured and clear text
                f"Generation {frame + 1}/{len(results)}\n"
                f"Best Path Distance: {distance:.2f}"
            )
        return line, text

    total_frames = len(results) + 30  # Adding 30 extra frames to pause at the end
    ani = FuncAnimation(fig, update, frames=total_frames, blit=True, interval=400)

    save_path = "../animations/genetic_algorithm_tsp/"

    if not os.path.exists(save_path):  # check if the directory exists
        os.makedirs(save_path)

    files_count = len(os.listdir(save_path))  # count the number of files in the directory for unique file name
    ani.save(f"{save_path}tsp_{files_count}.gif", writer="pillow", fps=10)  # Save the animation as gif


def main():
    result = list()  # list to store the results of the best individual in each generation
    cities = generate_random_cities(D, x_range=X_RANGE, y_range=Y_RANGE)  # generate random cities in the range
    # generate random population of individuals, each individual is a list of cities
    population = generate_random_population(cities, n=NP)

    for i in range(G):  # number of generations to run the algorithm
        new_population = copy.deepcopy(population)  # Offspring is always put to a new population
        for j in range(NP):  # iterate over each individual in the population
            parent_a = population[j]  # select the parent a
            parent_b = random.choice(population)  # select the parent b
            offspring_ab = mutate(order_crossover(parent_a, parent_b))  # crossover the parents and mutate the offspring

            # mutation probability
            if evaluate_individual(offspring_ab) < evaluate_individual(parent_a):
                new_population[j] = offspring_ab
        population = new_population
        best_individual = get_best_generation(population)
        result.append(best_individual)
        print(f"Generation {i + 1}/{G}, Best Path Distance: {evaluate_individual(best_individual):.2f}")

    animate(result, cities)  # save and visualize the results as gif


if __name__ == "__main__":
    main()
