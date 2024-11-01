import numpy as np

from solution import MAX_ITERATIONS, import_and_run


def blind_search(lower_bound, upper_bound, test_function, iterations=MAX_ITERATIONS):
    """Example blind search algorithm for optimization."""
    best_points_history = []
    best_points = []

    for _ in range(iterations):
        # Randomly select a point within the bounds
        x = np.random.uniform(lower_bound, upper_bound)
        y = np.random.uniform(lower_bound, upper_bound)
        point = [x, y]
        z = test_function(np.array(point))

        # Update the best points found so far
        if not best_points or z < test_function(np.array(best_points[-1])):
            best_points.append(point)
            best_points_history.append(list(best_points))  # Record the history of best points

    return best_points_history


if __name__ == '__main__':
    import_and_run(blind_search)
