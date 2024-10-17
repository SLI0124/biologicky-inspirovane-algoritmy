# all functions definition are taken from: https://www.sfu.ca/~ssurjano/optimization.html
import numpy as np


def function_ranges(function):
    if function == sphere:
        return -5.12, 5.12
    elif function == ackley:
        return -32.768, 32.768
    elif function == rastrigin:
        return -5.12, 5.12
    elif function == rosenbrock:
        return -5, 10
    elif function == griewank:
        return -5, 5
    elif function == schwefel:
        return -50, 50
    elif function == levy:
        return -10, 10
    elif function == michalewicz:
        return 0, np.pi
    elif function == zakharov:
        return -5, 10
    else:
        return Exception("Function not found")


def sphere(parameters):
    return np.sum(np.square(parameters))


def ackley(parameters):
    dimension = len(parameters)
    a = 20
    b = 0.2
    c = 2 * np.pi
    first_sum = np.sum(parameters ** 2)
    second_sum = np.sum(np.cos(c * parameters))
    return -a * np.exp(-b * np.sqrt(first_sum / dimension)) - np.exp(second_sum / dimension) + a + np.exp(1)


def rastrigin(parameters):
    dimension = len(parameters)
    first_sum = np.sum(parameters ** 2 - 10 * np.cos(2 * np.pi * parameters))
    return 10 * dimension + first_sum


def rosenbrock(parameters):
    dimension = len(parameters)
    total_sum = 0
    for i in range(dimension - 1):
        total_sum += 100 * (parameters[i + 1] - parameters[i] ** 2) ** 2 + (1 - parameters[i]) ** 2
    return total_sum


def griewank(parameters):
    dimension = len(parameters)
    first_sum = np.sum(parameters ** 2) / 4000
    second_sum = 1
    for i in range(dimension):
        second_sum *= np.cos(parameters[i] / np.sqrt(i + 1))
    return first_sum - second_sum + 1


def schwefel(parameters):
    dimension = len(parameters)
    constant = 418.9829 * dimension
    for i in range(dimension):
        parameters[i] = parameters[i] * np.sin(np.sqrt(np.abs(parameters[i])))
    return constant - np.sum(parameters)


def levy(parameters):
    dimension = len(parameters)
    w = 1 + (parameters - 1) / 4
    term1 = (np.sin(np.pi * w[0])) ** 2
    term3 = (w[-1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[-1])) ** 2)
    middle = 0
    for i in range(dimension - 1):
        wi = w[i]
        new = (wi - 1) ** 2 * (1 + 10 * (np.sin(np.pi * wi + 1)) ** 2)
        middle += new
    return term1 + middle + term3


def michalewicz(parameters):
    dimension = len(parameters)
    m = 10
    for i in range(dimension):
        parameters[i] = np.sin(parameters[i]) * np.sin((i + 1) * parameters[i] ** 2 / np.pi) ** (2 * m)
    return -np.sum(parameters)


def zakharov(parameters):
    first_sum = np.sum(parameters ** 2)
    second_sum = 0
    for i in range(len(parameters)):
        second_sum += 0.5 * (i + 1) * parameters[i]
    return first_sum + second_sum ** 2 + second_sum ** 4


def get_all_functions():
    return {
        1: sphere,
        2: ackley,
        3: rastrigin,
        4: rosenbrock,
        5: griewank,
        6: schwefel,
        7: levy,
        8: michalewicz,
        9: zakharov
    }
