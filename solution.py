import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from function import get_function_parameters, get_all_functions

import matplotlib as mpl

mpl.rcParams['figure.max_open_warning'] = 100  # Suppress warning for too many figures

DIMENSION = 2  # x and y to z for 3D plot
ALPHA_PLOT = 0.333  # Transparency of the surface plot
MAX_ITERATIONS = 33_333  # Maximum number of iterations for the optimization algorithms
# hill_climbing parameters
NUM_NEIGHBORS = 13  # Number of neighbors to generate
SIGMA_HILL = 0.666  # surrounding area for neighbors based on normal distribution
# simulated_annealing parameters
T_ZERO = 100  # initial temperature
T_MIN = 0.5  # minimum temperature
ALPHA_ANNEALING = 0.95  # temperature decrease factor
SIGMA_ANNEALING = 0.1  # standard deviation for normal distribution


class Solution:
    def __init__(self, dimension, lower_bound, upper_bound, step, function, algorithm, iterations):
        # Initialize parameters for the optimization problem
        self.d = dimension
        self.min = lower_bound
        self.max = upper_bound
        self.step = step
        self.f = function
        self.algorithm = algorithm
        self.iterations = iterations  # Number of iterations for the algorithm
        self.params = []  # Store parameters for the grid
        self.fig = plt.figure()  # Create a new figure
        self.ax = self.fig.add_subplot(111, projection='3d')  # Set up 3D plotting
        self.init_params()  # Initialize the parameter grid
        self.tmp = []  # To store the history of solutions
        self.points = []  # For animated plotting

    def init_params(self):
        """Initialize the grid for visualization based on the dimension."""
        for _ in range(self.d):
            self.params.append(np.arange(self.min, self.max, self.step))
        self.params = np.meshgrid(*self.params)

    def print_graph(self):
        """Print the surface of the function."""
        x = self.params[0]
        y = self.params[1]

        # Apply the function to each (x, y) pair in the grid
        z = np.zeros_like(x)
        for index in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[index, j] = self.f(np.array([x[index, j], y[index, j]]))

        self.ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', alpha=ALPHA_PLOT)

    def show_graph(self):
        """Display the graph of the function."""
        self.print_graph()
        plt.show()

    def find_minimum(self):
        """Run the search algorithm and store the history of best solutions."""
        best_value = float('inf')  # Start with the highest possible value
        self.tmp = []  # Reset temporary storage for points history

        # Execute the algorithm and collect points
        all_points = self.algorithm(self.min, self.max, self.f, self.iterations)  # Pass iterations

        for points in all_points:
            current_value = self.f(np.array(points[-1]))  # Get the function value at the last point
            if current_value < best_value:
                best_value = current_value
                self.tmp.append(points)  # Store only if this point is better

        return best_value  # Return the best solution found

    def animate(self, index):
        """Animate the best solutions found at each step, skipping empty frames."""
        print(f"Animating step #{index}")

        # Clear previous points from the plot
        for point in self.points:
            point.set_data([], [])
            point.set_3d_properties([])

        self.points = []  # Reset points for the current animation frame

        # Plot new points for the current iteration, only if they exist
        if index < len(self.tmp) and self.tmp[index]:  # Ensure there are points to plot
            for p in self.tmp[index]:
                z_value = self.f(np.array(p))
                if np.isfinite(z_value):  # Ensure z_value is valid
                    point_plot, = self.ax.plot([p[0]], [p[1]], [z_value], 'ro', markersize=10)  # plot a single point
                    self.points.append(point_plot)

        return self.points  # Return the points for the animation

    def save_anim(self):
        """Save the animation of the optimization process as a GIF file."""
        self.print_graph()  # Print the surface graph
        anim = FuncAnimation(self.fig, self.animate, frames=min(len(self.tmp), 50), interval=250)
        output_path = f"animations/{self.algorithm.__name__}/{self.f.__name__}.gif"

        # Create the output directory if it doesn't exist
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        anim.save(output_path, writer='pillow')
        plt.close(self.fig)  # Close the figure after saving


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


def simulated_annealing(lower_bound, upper_bound, test_function, iterations=MAX_ITERATIONS, t_zero=T_ZERO,
                        t_min=T_MIN, alpha=ALPHA_ANNEALING, initial_sigma=SIGMA_ANNEALING):
    """Improved simulated annealing algorithm for optimization."""
    best_points_history = []

    x0 = [np.random.uniform(lower_bound, upper_bound) for _ in range(2)]  # Start from a random point
    fitness = test_function(np.array(x0))  # Get the value of the initial point
    best_points_history.append([list(x0)])  # Record the initial point
    best_fitness = fitness  # Track the best fitness
    best_solution = list(x0)  # Track the best solution

    # Perform iterations of simulated annealing
    acceptance_count = 0  # Count accepted worse solutions
    for i in range(iterations):
        t = max(t_zero * (alpha ** (i // 100)), t_min)  # Temperature decay every 100 iterations
        sigma = initial_sigma * (1 - (i / iterations))  # Decrease sigma over time

        # Generate a random neighbor
        x1 = [np.random.normal(x0[j], sigma) for j in range(2)]
        # Clip to bounds
        x1 = np.clip(x1, lower_bound, upper_bound)
        fitness1 = test_function(np.array(x1))  # Evaluate the function at the neighbor

        # Acceptance criteria
        if fitness1 < fitness or np.random.uniform(0, 1) < np.exp((fitness - fitness1) / t):
            x0 = x1  # Update the current point
            fitness = fitness1  # Update the fitness value
            acceptance_count += 1  # Count the acceptance

            if fitness < best_fitness:  # Update the best found solution
                best_fitness = fitness
                best_solution = list(x0)

        best_points_history.append([list(x0)])  # Record the current best point

    print(f"Acceptance rate: {acceptance_count / iterations:.2f}")
    return best_points_history


if __name__ == '__main__':

    matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib

    if DIMENSION != 2:
        raise Exception("Function requires 2 dimensions.")

    # Iterate through all available functions and apply the optimization
    for i, function in get_all_functions().items():
        lower_bound, upper_bound, step_size = get_function_parameters(function)

        # blind search algorithm
        solution_blind_search = Solution(DIMENSION, lower_bound, upper_bound, step_size, function, blind_search,
                                         MAX_ITERATIONS)
        best_solution_blind_search = solution_blind_search.find_minimum()
        print(f"Function: {function.__name__}, Algorithm: blind_search, "
              f"Best found solution: {best_solution_blind_search}")
        solution_blind_search.save_anim()

        # hill climbing algorithm
        solution_hill_climbing = Solution(DIMENSION, lower_bound, upper_bound, step_size, function, hill_climbing,
                                          MAX_ITERATIONS)
        best_solution_hill_climbing = solution_hill_climbing.find_minimum()
        print(f"Function: {function.__name__}, Algorithm: hill_climbing, "
              f"Best found solution: {best_solution_hill_climbing}")
        solution_hill_climbing.save_anim()

        # simulated annealing algorithm
        solution_simulated_annealing = Solution(DIMENSION, lower_bound, upper_bound, step_size, function,
                                                simulated_annealing, MAX_ITERATIONS)
        best_solution_simulated_annealing = solution_simulated_annealing.find_minimum()
        print(f"Function: {function.__name__}, Algorithm: simulated_annealing, "
              f"Best found solution: {best_solution_simulated_annealing}")
        solution_simulated_annealing.save_anim()
