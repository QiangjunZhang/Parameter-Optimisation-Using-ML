import os
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.misc import derivative
from deap import creator, base, tools, algorithms
import random

DENSITY = 1100                           # material density
START = 25                               # normalized start temperature
END = 725                                # normalized end temperature


def load_data(filepath, skip):
    if filepath[-1] == 'x':
        data = pd.read_excel(filepath, skiprows=skip, header=None)
    else:
        data = pd.read_csv(filepath, skiprows=skip, header=None)
    return data


# export mass, time, temp by its collum number
def export(data, mass_col, time_col, temp_col, minute=True, mass_unit=True):
    raw_mass = data.iloc[:, mass_col].values
    if mass_unit:
        raw_mass /= 100
    raw_time = data.iloc[:, time_col].values
    if minute:
        raw_time *= 60
    raw_temp = data.iloc[:, temp_col].values
    return raw_mass, raw_time, raw_temp


# Normalize the data to fixed x axis range and interval by smooth and interpolation
def preprocess(raw_mass, raw_temp):
    raw_mass *= DENSITY
    curve = UnivariateSpline(raw_temp, raw_mass)
    norm_temp = np.linspace(START, END, END - START, endpoint=True)
    # norm_mass = curve(norm_temp)
    massLossRate = -derivative(curve, norm_temp)
    return norm_temp, massLossRate


# writing data into csv
def write_data(norm_temp, massLossRate, filename):
    exp = np.asarray([norm_temp, massLossRate]).T
    exp = np.insert(exp, 0, [0, massLossRate[0]], axis=0)
    df = pd.DataFrame(exp)
    collum_name = ['Temp', 'Mass Loss Rate']
    df.to_csv(filename + 'massLossRate.csv', index=False, header=collum_name, float_format='%.3f')


# generate normalized testing data for learning
def normalized_test_data(filename):
    skip_header = 1
    data = load_data(filename, skip_header)
    mass_col, time_col, temp_col = 2, 0, 3
    raw_mass, raw_time, raw_temp = export(data, mass_col, time_col, temp_col, minute=True, mass_unit=True)
    norm_temp, massLossRate = preprocess(raw_mass, raw_temp)
    write_data(norm_temp, massLossRate, filename)
    return norm_temp, massLossRate


# call the simulation program and normalize the predicted data for comparison
def call_simulation():
    os.system("gpyro.exe gpyro.data")
    filename = 'gpyro_summary_01_0001.csv'
    skip_header = 2
    data = load_data(filename, skip_header)
    mass_col, time_col, temp_col = 2, 3, 1
    raw_mass, raw_time, raw_temp = export(data, mass_col, time_col, temp_col, minute=False, mass_unit=False)
    norm_temp, massLossRate = preprocess(raw_mass, raw_temp)
    return norm_temp, massLossRate


# modify the simulation parameters in the input file
def update_input(Z1, E1, N1, Z2, E2, N2, R2):
    file = open('gpyro.data', 'r')
    data = file.readlines()
    file.close()
    data[171] = 'Z(1) =  ' + str(Z1).strip('[]') + '\n'
    data[172] = 'E(1) =  ' + str(E1).strip('[]') + ',\n'
    data[176] = 'ORDER(1) =  ' + str(N1).strip('[]') + ',\n'
    data[185] = 'Z(2) =  ' + str(Z2).strip('[]') + '\n'
    data[186] = 'E(2) =  ' + str(E2).strip('[]') + ',\n'
    data[190] = 'ORDER(2) =  ' + str(N2).strip('[]') + ',\n'
    data[124] = 'R0(2) =  ' + str(R2) + ',\n'
    file = open('gpyro.data', 'w')
    file.writelines(data)
    file.close()


fitness = []
norm_temp, norm_mlr = normalized_test_data('test_data.xlsx')
output = open('output.txt', 'w')


# genetic algorithm optimiser
def gene_generator():
    x = random.random() * 8
    return round(x, 2)


# fitness evaluation
def evaluate(individual):
    Z1 = np.power(10, individual[0]) * 10e5
    E1 = individual[1] * 7 + 150
    N1 = individual[2] * 0.1 + 1
    Z2 = np.power(10, individual[3]) * 10e7
    E2 = individual[4] * 5 + 190
    N2 = individual[5] * 0.1 + 1.6
    R2 = individual[6] * 5 + 395
    update_input(Z1, E1, N1, Z2, E2, N2, R2)
    predicted_temp, predicted_mlr = call_simulation()
    fitness.append(np.sum(np.square(predicted_temp - norm_temp)) + np.square(predicted_mlr.max() - norm_mlr.max()))
    output.write(str(individual).replace('[', '').replace(']', '') + ',' + str(fitness[-1]) + '\n')
    return 1 / fitness[-1]


# GA optimisation
def optimizer():

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual, gene_generator, 7)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=5, sigma=5, indpb=0.01)
    toolbox.register("select", tools.selTournament, tournsize=50)
    population = toolbox.population(n=1000)

    NGEN = 100
    for _ in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    output.close()
    print(fitness)


if __name__ == '__main__':
    optimizer()
