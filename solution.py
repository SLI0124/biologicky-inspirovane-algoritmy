import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from function import function_ranges, get_all_functions


class Solution:
    def __init__(self, dimension, lower_bound, upper_bound, step, function, algorithm):
        # Initialize parameters for the optimization problem
        self.d = dimension
        self.min = lower_bound
        self.max = upper_bound
        self.step = step
        self.f = function
        self.algorithm = algorithm
        self.params = []
        self.fig = plt.figure()
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
        if self.d != 2:
            raise Exception("Function requires 2 dimensions.")

        x = self.params[0]
        y = self.params[1]

        # Apply the function to each (x, y) pair in the grid, this always had some issues
        z = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i, j] = self.f(np.array([x[i, j], y[i, j]]))

        self.ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', alpha=0.7)

    def show_graph(self):
        """Display the graph of the function."""
        if self.d != 2:
            raise Exception("Function requires 2 dimensions.")
        self.print_graph()
        plt.show()

    def find_minimum(self, aargs=()):
        """Run the search algorithm and store the history of best solutions."""
        best_value = float('inf')  # Start with a high initial value
        self.tmp = []  # Reset temporary storage for points history

        # Execute the algorithm and collect points
        all_points = self.algorithm(self.d, self.min, self.max, self.f, *aargs)

        for points in all_points:
            current_value = self.f(np.array(points[-1]))  # Get the function value at the last point
            if current_value < best_value:
                best_value = current_value
                self.tmp.append(points)  # Store only if this point is better

        return best_value  # Return the best solution found

    def animate(self, i):
        """Animate the best solutions found at each step, skipping empty frames."""
        if self.d != 2:
            raise Exception("Function requires 2 dimensions.")

        print(f"Animating step #{i}")

        # Clear previous points from the plot
        for point in self.points:
            point.set_data([], [])
            point.set_3d_properties([])

        self.points = []  # Reset points for the current animation frame

        # Plot new points for the current iteration, only if they exist
        if i < len(self.tmp) and self.tmp[i]:  # Ensure there are points to plot
            for p in self.tmp[i]:
                z_value = self.f(np.array(p))
                if np.isfinite(z_value):  # Ensure z_value is valid
                    point_plot, = self.ax.plot([p[0]], [p[1]], [z_value], 'ro')  # Plot single point
                    self.points.append(point_plot)

        # Only display the current frame if there are points plotted
        if self.points:
            print(f"Completed animation for step #{i}")
        else:
            print(f"No points to animate for step #{i}, skipping.")

    def save_anim(self):
        """Save the animation of the optimization process as a GIF file."""
        if self.d != 2:
            raise Exception("Function requires 2 dimensions.")

        self.print_graph()
        anim = FuncAnimation(self.fig, self.animate, frames=min(len(self.tmp), 50), interval=500)
        output_path = f"animations/{self.algorithm.__name__}_{self.f.__name__}.gif"

        # Create the output directory if it doesn't exist
        if not os.path.exists('animations'):
            os.makedirs('animations')

        print(f"Saving animation to: {output_path}")
        anim.save(output_path, writer='pillow')


def blind_search(dimension, lower_bound, upper_bound, test_function, iterations=100):
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
    import matplotlib

    matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib

    dimension = 2  # We need 2D for plotting
    step_size = 0.1
    max_iterations = 100  # Adjust this by your needs

    # Iterate through all available functions and apply the optimization
    for i, function in get_all_functions().items():
        lower_bound, upper_bound = function_ranges(function)

        solution = Solution(dimension, lower_bound, upper_bound, step_size, function, blind_search)

        # Find the minimum (and store history of best points)
        best_solution = solution.find_minimum(aargs=(max_iterations,))
        print(f"Function: {function.__name__}, Best found solution: {best_solution}")

        # Save the animation as a GIF
        solution.save_anim()
