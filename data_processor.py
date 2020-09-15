import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.misc import derivative


class DataProcessor:

    def __init__(self, density=1100, start=25, end=725):
        self.DENSITY = density  # material density
        self.START = start      # normalized start temperature
        self.END = end          # normalized end temperature

    def load_data(self, filepath, skip):
        if filepath[-1] == 'x':
            data = pd.read_excel(filepath, skiprows=skip, header=None)
        else:
            data = pd.read_csv(filepath, skiprows=skip, header=None)
        return data

    # export mass, time, temp by its collum number
    def export(self, data, mass_col, time_col, temp_col, minute=True, mass_unit=True):
        raw_mass = data.iloc[:, mass_col].values
        if mass_unit:
            raw_mass /= 100
        raw_time = data.iloc[:, time_col].values
        if minute:
            raw_time *= 60
        raw_temp = data.iloc[:, temp_col].values
        return raw_mass, raw_time, raw_temp

    # Normalize the data to fixed x axis range and interval by smooth and interpolation
    def preprocess(self, raw_mass, raw_temp):
        raw_mass *= self.DENSITY
        curve = UnivariateSpline(raw_temp, raw_mass)
        norm_temp = np.linspace(self.START, self.END, self.END - self.START, endpoint=True)
        massLossRate = -derivative(curve, norm_temp)
        return norm_temp, massLossRate

    # writing data into csv
    def write_data(self, norm_temp, massLossRate, filename):
        exp = np.asarray([norm_temp, massLossRate]).T
        exp = np.insert(exp, 0, [0, massLossRate[0]], axis=0)
        df = pd.DataFrame(exp)
        collum_name = ['Temp', 'Mass Loss Rate']
        df.to_csv(filename + 'massLossRate.csv', index=False, header=collum_name, float_format='%.3f')

    # generate normalized testing data for learning
    def normalized_test_data(self, filename):
        skip_header = 1
        data = self.load_data(filename, skip_header)
        mass_col, time_col, temp_col = 2, 0, 3
        raw_mass, raw_time, raw_temp = self.export(data, mass_col, time_col, temp_col, minute=True, mass_unit=True)
        norm_temp, massLossRate = self.preprocess(raw_mass, raw_temp)
        self.write_data(norm_temp, massLossRate, filename)
        return norm_temp, massLossRate
