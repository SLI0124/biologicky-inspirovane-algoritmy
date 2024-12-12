import numpy as np

from solution import T_ZERO, T_MIN, ALPHA_ANNEALING, import_and_run


def simulated_annealing(lower_bound, upper_bound, test_function, t_zero=T_ZERO,
                        t_min=T_MIN, alpha=ALPHA_ANNEALING):
    """Simulated annealing algorithm for optimization."""
    best_points_history = []
    current_point = [np.random.uniform(lower_bound, upper_bound) for _ in range(2)]  # Start from a random point
    current_fitness = test_function(np.array(current_point))  # Get the value of the initial point
    best_points_history.append([list(current_point)])

    temperature = t_zero  # Initial temperature
    while temperature > t_min:  # Continue until the temperature reaches the minimum value
        neighbor = [np.random.normal(current_point[0], 1), np.random.normal(current_point[1], 1)]  # Generate a neighbor
        # Ensure the neighbor is within the bounds
        neighbor = [min(max(neighbor[0], lower_bound), upper_bound), min(max(neighbor[1], lower_bound), upper_bound)]
        neighbor_fitness = test_function(np.array(neighbor))  # Evaluate the function at the neighbor
        if neighbor_fitness < current_fitness:  # Update if the neighbor has a better fitness value
            current_point = neighbor
            current_fitness = neighbor_fitness
        else:
            acceptance_probability = np.exp(-(neighbor_fitness - current_fitness) / temperature)
            if np.random.uniform(0, 1) < acceptance_probability:  # Accept the new solution with a probability
                current_point = neighbor
        temperature *= alpha  # Decrease the temperature

        best_points_history.append([list(current_point)])  # Record the best point after each iteration

    return best_points_history


if __name__ == '__main__':
    import_and_run(simulated_annealing)
