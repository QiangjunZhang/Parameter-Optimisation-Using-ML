import os
import numpy as np
from data_processor import DataProcessor


class AbstractOptimiser(DataProcessor):

    def __init__(self, test_file, density=1100, start=25, end=72):
        super().__init__(density, start, end)
        self._fitness = []
        self.norm_temp, self.norm_mlr = self.normalized_test_data(test_file)

    # call the simulation program and normalize the predicted data for comparison
    def call_simulation(self):
        simulation_command = "gpyro.exe gpyro.data"
        output_file_name = 'gpyro_summary_01_0001.csv'
        os.system(simulation_command)
        skip_header = 2
        data = self.load_data(output_file_name, skip_header)
        mass_col, time_col, temp_col = 2, 3, 1
        raw_mass, raw_time, raw_temp = self.export(data, mass_col,
                                                   time_col, temp_col,
                                                   minute=False, mass_unit=False)
        return self.preprocess(raw_mass, raw_temp)

    # modify the simulation parameters in the input file
    def update_input(self, *args):
        Z1, E1, N1, Z2, E2, N2, R2 = args
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

    # fitness evaluation
    def evaluate(self, *args):
        self.update_input(*args)
        predicted_temp, predicted_mlr = self.call_simulation()
        self._fitness.append(np.sum(np.square(predicted_temp - self.norm_temp))
                             + np.square(predicted_mlr.max() - self.norm_mlr.max()))
        return 1 / self._fitness[-1]

    def get_fitness(self):
        return self._fitness

    def reset_fitness(self):
        self._fitness = []
