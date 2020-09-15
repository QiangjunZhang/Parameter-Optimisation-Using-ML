from ga_optimiser import OptimiserGA
from fmin_optimiser import FminOptimiser


def test_ga_optimiser():
    optimiser = OptimiserGA('test_data.xlsx')
    gene_num = 7
    max_generation = 500
    population = 1000
    optimiser.optimise(gene_num, max_generation, population)
    print(optimiser.get_fitness())


def test_fmin_optimiser():
    optimiser = FminOptimiser('test_data.xlsx')
    initial_parameters = (5.0, 199.8, 1, 1.01, 211.76, 1.5749698, 408.79)
    optimiser.optimise_fmin(initial_parameters)
    print(optimiser.get_fitness())


if __name__ == '__main__':
    test_ga_optimiser()
    test_fmin_optimiser()