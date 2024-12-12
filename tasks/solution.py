import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from tasks.function import get_function_parameters, get_all_functions

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


class Solution:
    def __init__(self, dimension, lower_bound, upper_bound, step, function, algorithm):
        # Initialize parameters for the optimization problem
        self.d = dimension
        self.min = lower_bound
        self.max = upper_bound
        self.step = step
        self.f = function
        self.algorithm = algorithm
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
        all_points = self.algorithm(self.min, self.max, self.f)  # Pass iterations

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
        output_path = f"../animations/{self.algorithm.__name__}/{self.f.__name__}.gif"

        # Create the output directory if it doesn't exist
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        anim.save(output_path, writer='pillow')
        plt.close(self.fig)  # Close the figure after saving


def import_and_run(algorithm):
    matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib

    if DIMENSION != 2:
        raise Exception("Function requires 2 dimensions.")

    # Iterate through all available functions and apply the optimization
    for i, function in get_all_functions().items():
        lower_bound, upper_bound, step_size = get_function_parameters(function)

        solution = Solution(DIMENSION, lower_bound, upper_bound, step_size, function, algorithm)
        best_solution = solution.find_minimum()
        function_name = function.__name__ if hasattr(function, '__name__') else str(function)
        algorithm_name = algorithm.__name__ if hasattr(algorithm, '__name__') else str(algorithm)
        print(f"Function: {function_name}, Algorithm: {algorithm_name}, Best found solution: {best_solution}")
        solution.save_anim()
