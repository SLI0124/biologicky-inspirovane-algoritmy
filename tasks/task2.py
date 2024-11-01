import numpy as np

from solution import MAX_ITERATIONS, SIGMA_HILL, NUM_NEIGHBORS, import_and_run


def hill_climbing(lower_bound, upper_bound, test_function, iterations=MAX_ITERATIONS, num_neighbors=NUM_NEIGHBORS,
                  sigma=SIGMA_HILL):
    """Hill climbing algorithm for optimization with neighbor generation using normal distribution."""

    def generate_neighbors(x):
        """Generate neighbors using normal distribution."""
        neighbors = []
        normals = []

        # Generate normal distributions for each dimension
        for i in range(len(x)):
            normals.append(np.random.normal(x[i], sigma, num_neighbors))  # make num_neighbors normal values around x[i]

        # Combine normal values to create num_neighbors neighbors
        for i in range(num_neighbors):
            p = []
            for j in range(len(x)):
                p.append(normals[j][i])
            neighbors.append(p)

        return neighbors

    best_points_history = []
    x0 = [np.random.uniform(lower_bound, upper_bound) for _ in range(2)]  # Start from a random point
    fitness = test_function(np.array(x0))  # Get the value of the initial point
    best_points_history.append([list(x0)])

    # Perform iterations of hill climbing
    for _ in range(iterations):
        neighbors = generate_neighbors(x0)  # Generate neighbors
        for nb in neighbors:
            val = test_function(np.array(nb))  # Evaluate the function at each neighbor
            if val < fitness:  # Update if the neighbor has a better fitness value
                x0 = nb
                fitness = val
        best_points_history.append([list(x0)])  # Record the best point after each iteration

    return best_points_history


if __name__ == '__main__':
    import_and_run(hill_climbing)
