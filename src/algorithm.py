import random
import math
from config import *
from functools import cmp_to_key


def compare_members(ind1, ind2):
    if math.fabs(fitness_function(ind1)[0] - config['global_minimum']) > math.fabs(fitness_function(ind2)[0] - config['global_minimum']):
        return 1
    if math.fabs(fitness_function(ind1)[0] - config['global_minimum']) == math.fabs(fitness_function(ind2)[0] - config['global_minimum']):
        return 0
    else:
        return -1


def create_binary_individual(icls):
    genome = list()
    for _ in range(40):
        genome.append(random.randint(0, 1))

    return icls(genome)


def create_real_individual(icls):
    genome = list()
    genome.append(random.uniform(config['interval'][0], config['interval'][1]))
    genome.append(random.uniform(config['interval'][0], config['interval'][1]))

    return icls(genome)


def calculate_binary_arr_value(binary_arr):
    binary_string = ''.join([str(elem) for elem in binary_arr])
    return config['interval'][0] + int(binary_string, 2) * (config['interval'][1] - config['interval'][0]) / (math.pow(2, len(binary_arr)) - 1)


def decode_individual(individual):
    if len(individual) > 2:
        return calculate_binary_arr_value(individual[20:]), calculate_binary_arr_value(individual[:20])
    else:
        return individual[0], individual[1]


def is_value_in_interval(value):
    return config['interval'][0] <= value <= config['interval'][1]


def fitness_function(individual):
    x_1, x_2 = decode_individual(individual)
    return (x_1 + 2 * x_2 - 7) ** 2 + (2 * x_1 + x_2 - 5) ** 2,


def arithmetic_crossover(ind1, ind2):
    probability = random.random()

    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        ind1[i] = probability * x1 + (1 - probability) * x2
        ind2[i] = (1 - probability) * x1 + probability * x2

    return ind1, ind2


def blend_crossover(ind1, ind2, alpha, beta=None):
    beta = beta if beta is not None else alpha

    for ind in [ind1, ind2]:
        dx = abs(ind1[0] - ind2[0])
        dy = abs(ind1[1] - ind2[1])

        while True:
            ind[0] = random.uniform(
                min(ind1[0], ind2[0]) - alpha * dx,
                max(ind1[0], ind2[0]) + beta * dx)
            ind[1] = random.uniform(
                min(ind1[1], ind2[1]) - alpha * dy,
                max(ind1[1], ind2[1]) + beta * dy)

            if is_value_in_interval(ind[0]) and is_value_in_interval(ind[1]):
                break

    return ind1, ind2


def linear_crossover(ind1, ind2):
    factors = [
        [[0.5, 0.5], [0.5, 0.5]],
        [[1.5, -0.5], [1.5, -0.5]],
        [[-0.5, 1.5], [-0.5, 1.5]]
    ]
    temp_values = [[], [], []]

    for i in range(3):
        for j in range(len(ind1)):
            temp_values[i].append(factors[i][j][0] * ind1[j] + factors[i][j][1] * ind2[j])

    result = sorted(temp_values, key=cmp_to_key(compare_members),
                    reverse=OptimizationType.MAXIMIZATION == config['optimization_type'])

    return result[:2]


def average_crossover(ind1, ind2):
    for idx in range(len(ind1)):
        ind1[idx] = (ind1[idx] + ind2[idx]) / 2

    return ind1, ind2
