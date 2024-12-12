import numpy as np
from task5 import generate_population
from tasks.solution import import_and_run

# Constants
POP_SIZE = 20
M_MAX = 500
C1, C2 = 2, 2
V_MAX, V_MIN = 1, -1
W_MAX, W_MIN = 0.9, 0.4
DIMENSION = 2


def update_velocity(v, x, p_best, g_best, w, c1, c2, v_min, v_max):
    """Update velocity of a particle."""
    r1, r2 = np.random.rand(), np.random.rand()  # x  Random numbers for stochasticity
    new_v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)  # Update velocity based on formula
    return np.clip(new_v, v_min, v_max)  # Clip velocity to bounds


def update_position(x, v, lower_bound, upper_bound):
    """Update position of a particle."""
    new_x = x + v  # Update position based on velocity
    return np.clip(new_x, lower_bound, upper_bound)  # Clip position to bounds


def get_best_position(population, test_function):
    """Get the best position in the population."""
    best_position = population[0]  # Initialize best position to first individual
    best_value = test_function(
        population[0])  # Initialize best value to first individual's value based on test function
    for individual in population:  # Iterate over all individuals in the population
        value = test_function(individual)  # Calculate value of individual based on test function
        if value < best_value:  # Update best value and position if current individual is better
            best_value = value  # Update best value
            best_position = individual  # Update best position
    return best_position


def particle_swarm_optimization(lower_bound, upper_bound, test_function, pop_size=POP_SIZE, dimension=DIMENSION,
                                m_max=M_MAX, c1=C1, c2=C2, v_max=V_MAX, v_min=V_MIN, w_max=W_MAX, w_min=W_MIN):
    # Initialize swarm, best positions, and velocities
    swarm = np.array(generate_population(lower_bound, upper_bound, pop_size, dimension))  # Generate initial population
    p_best = np.copy(swarm)  # Initialize personal best positions
    g_best = get_best_position(swarm, test_function)  # Initialize global best position
    v = np.array(generate_population(v_min, v_max, pop_size, dimension))  # Initialize velocities

    all_points = []
    for m in range(m_max):  # Iterate over all generations based on max iterations
        iteration_points = []
        w = w_max - (w_max - w_min) * (m / m_max)

        for i in range(pop_size):  # Iterate over all individuals in the population
            x, p_best_i = swarm[i], p_best[i]  # Get current position and personal best position
            v[i] = update_velocity(v[i], x, p_best_i, g_best, w, c1, c2, v_min, v_max)  # Update velocity
            swarm[i] = update_position(x, v[i], lower_bound, upper_bound)  # Update position

            # Update personal and global bests
            if test_function(swarm[i]) < test_function(p_best_i):  # Update personal best if current position is better
                p_best[i] = swarm[i]  # Update personal best
                if test_function(p_best[i]) < test_function(g_best):  # Update global best if personal best is better
                    g_best = p_best[i]  # Update global best

            iteration_points.append(swarm[i].tolist())
        all_points.append(iteration_points)

    return all_points


if __name__ == '__main__':
    import_and_run(particle_swarm_optimization)
