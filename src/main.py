from deap import base
from deap import creator
from deap import tools
from config import *
from algorithm import *
import random
import csv
import sys
import os
import matplotlib.pyplot as plt

best_members = []
avg_arr = []
std_arr = []

sub_folder = sys.argv[0][
         :-7] + f'output/{config["individual"].value}/{config["optimization_type"].value}_S={config["select"].value}_C={config["mate"].value}_M={config["mutate"].value}/'
if not os.path.isdir(sub_folder):
    os.mkdir(sub_folder)

output_file_path = sub_folder + 'output.csv'
plot_path_fv = sub_folder + 'fitness_value.png'
plot_path_avg = sub_folder + 'average.png'
plot_path_stdev = sub_folder + 'standard_deviation.png'


def write_to_file(best, avg, std):
    header = ['Epoch', 'X1', 'X2', 'Fitness_value', 'Average', 'Standard_deviation']

    with open(output_file_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        for epoch in range(len(best)):
            writer.writerow([
                epoch + 1,
                best[epoch][0],
                best[epoch][1],
                fitness_function(best[epoch])[0],
                avg[epoch],
                std[epoch]
            ])


def create_plots(best, avg, std):
    x = range(1, config['number_iteration'] + 1)

    plt.xlabel('Epoch')
    plt.ylabel('Fitness value')
    plt.title("Chart of fitness value by epoch")

    plt.plot(x, [fitness_function(ind)[0] for ind in best])
    plt.savefig(plot_path_fv)
    plt.cla()

    plt.xlabel('Epoch')
    plt.ylabel('Average')
    plt.title("Chart of average by epoch")

    plt.plot(x, [a for a in avg])
    plt.savefig(plot_path_avg)
    plt.cla()

    plt.xlabel('Epoch')
    plt.ylabel('Standard deviation')
    plt.title("Chart of standard deviation by epoch")

    plt.plot(x, [s for s in std])
    plt.savefig(plot_path_stdev)
    plt.cla()


if __name__ == '__main__':
    match config['optimization_type']:
        case OptimizationType.MAXIMIZATION:
            creator.create("Fitness", base.Fitness, weights=(1.0,))
        case _:
            creator.create("Fitness", base.Fitness, weights=(-1.0,))

    creator.create("Individual", list, fitness=creator.Fitness)
    toolbox = base.Toolbox()

    match config['individual']:
        case Individual.BINARY:
            toolbox.register('individual', create_binary_individual, creator.Individual)
        case Individual.REAL:
            toolbox.register('individual', create_real_individual, creator.Individual)
        case _:
            print("Bad config")
            exit(-1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_function)

    match config['select']:
        case Selection.TOURNAMENT:
            toolbox.register("select", tools.selTournament, tournsize=config['tournament_size'])
        case Selection.WORST:
            toolbox.register("select", tools.selWorst, k=config['select_size'])
        case Selection.RANDOM:
            toolbox.register("select", tools.selRandom, k=config['select_size'])
        case Selection.ROULETTE_WHEEL:
            toolbox.register("select", tools.selRoulette, k=config['select_size'])
        case _:
            toolbox.register("select", tools.selBest, k=config['select_size'])

    if Individual.BINARY == config['individual']:
        match config['mate']:

            case Crossover.ONE_POINT:
                toolbox.register("mate", tools.cxOnePoint)
            case Crossover.UNIFORM:
                toolbox.register("mate", tools.cxUniform, indpb=1)
            case Crossover.ORDERED:
                toolbox.register("mate", tools.cxOrdered)
            case Crossover.TWO_POINT:
                toolbox.register("mate", tools.cxTwoPoint)
            case _:
                print("Bad config")
                exit(-1)
    else:
        match config['mate']:
            case Crossover.BLEND:
                toolbox.register("mate", tools.cxBlend, alpha=config['alpha'])
            case Crossover.ARITHMETIC:
                toolbox.register("mate", arithmetic_crossover)
            case Crossover.BLEND_ALPHA:
                toolbox.register("mate", blend_crossover, alpha=config['alpha'])
            case Crossover.BLEND_ALPHA_BETA:
                toolbox.register("mate", blend_crossover, alpha=config['alpha'], beta=config['beta'])
            case Crossover.AVERAGE:
                toolbox.register("mate", average_crossover)
            case Crossover.LINEAR:
                toolbox.register("mate", linear_crossover)
            case Crossover.SIMULATED_BINARY:
                toolbox.register("mate", tools.cxSimulatedBinary, eta=1)
            case _:
                print("Bad config")
                exit(-1)
    if Individual.BINARY == config['individual']:
        match config['mutate']:
            case Mutation.FLIP_BIT:
                toolbox.register("mutate", tools.mutFlipBit, indpb=1)
            case Mutation.SHUFFLE_INDEXES:
                toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1)
            case _:
                print("Bad config")
                exit(-1)
    else:
        match config['mutate']:
            case Mutation.GAUSSIAN:
                toolbox.register("mutate", tools.mutGaussian, mu=config['mu'], sigma=config['sigma'], indpb=1)
            case Mutation.UNIFORM_INT:
                toolbox.register("mutate", tools.mutUniformInt, low=config['low'], up=config['up'], indpb=1)
            case _:
                print("Bad config")
                exit(-1)

    pop = toolbox.population(n=config['size_population'])
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    g = 0
    while g < config['number_iteration']:
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        listElitism = []
        for x in range(0, config['number_elite']):
            listElitism.append(tools.selBest(pop, 1)[0])

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < config['probability_mate']:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < config['probability_mutate']:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))
        pop[:] = offspring + listElitism
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        best_ind = tools.selBest(pop, 1)[0]

        best_members.append(decode_individual(best_ind))
        avg_arr.append(mean)
        std_arr.append(std)
        print("Best individual is %s, %s" % (decode_individual(best_ind), best_ind.fitness.values[0]))
    print("-- End of (successful) evolution --")
    write_to_file(best_members, avg_arr, std_arr)
    create_plots(best_members, avg_arr, std_arr)
