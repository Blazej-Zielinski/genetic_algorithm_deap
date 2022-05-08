from enum import Enum


class OptimizationType(Enum):
    MAXIMIZATION = 'FitnessMax'
    MINIMIZATION = 'FitnessMin'


class Individual(Enum):
    REAL = 'REAL'
    BINARY = 'BINARY'


class Selection(Enum):
    BEST = 'selBest'
    ROULETTE_WHEEL = 'selRoulette'
    WORST = 'selWorst'
    TOURNAMENT = 'selTournament'
    RANDOM = 'selRandom'
    # Additional
    DOUBLE_TOURNAMENT = 'selDoubleTournament'
    STOCH_UNIVERSAL_SAMPLING = 'selStochasticUniversalSampling'


class Crossover(Enum):
    # BINARY
    UNIFORM = "cxUniform"
    ONE_POINT = 'cxOnePoint'
    TWO_POINT = 'cxTwoPoint'
    # Additional
    ORDERED = 'cxOrdered'

    # REAL
    ARITHMETIC = 'arithmetic'
    BLEND_ALPHA = 'blend-alpha'
    BLEND_ALPHA_BETA = 'blend-alpha-beta'
    AVERAGE = 'average'
    LINEAR = 'linear'
    # Additional
    BLEND = 'cxBlend'
    SIMULATED_BINARY = 'cxSimulatedBinary'


class Mutation(Enum):
    # BINARY
    SHUFFLE_INDEXES = 'mutShuffleIndexes'
    FLIP_BIT = 'mutFlipBit'

    # REAL
    GAUSSIAN = 'mutGaussian'
    UNIFORM_INT = 'mutUniformInt'


config = {
    'individual': Individual.REAL,
    'optimization_type': OptimizationType.MINIMIZATION,
    'select': Selection.BEST,
    'mate': Crossover.LINEAR,
    'mutate': Mutation.UNIFORM_INT,

    'interval': [-10, 10],
    'size_population': 100,
    'select_size': 100,
    'tournament_size': 60,
    'probability_mate': 0.6,
    'probability_mutate': 0.1,
    'number_elite': 10,

    'number_iteration': 1000,

    # Additional parameters
    'global_minimum': 0,

    # GAUSSIAN
    'mu': 0,
    'sigma': 1,

    # UNIFORM_INT
    'low': -10,
    'up': 10,

    # BLEND CROSSOVER
    'alpha': 0.6,
    'beta': 0.2
}
