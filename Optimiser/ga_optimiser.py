from deap import creator, base, tools, algorithms
import random
from Optimiser.base_optimiser import AbstractOptimiser


class OptimiserGA(AbstractOptimiser):

    # genetic algorithm optimiser
    def gene_generator(self, num):
        return round(random.random() * num, 2)

    # GA optimisation
    def optimise(self, gene_num, max_generation, population):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initRepeat, creator.Individual, self.gene_generator(gene_num), 7)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=5, sigma=5, indpb=0.01)
        toolbox.register("select", tools.selTournament, tournsize=50)
        population = toolbox.population(n=population)

        for _ in range(max_generation):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=len(population))
