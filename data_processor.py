import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
from scipy.misc import derivative
from matplotlib import pyplot as plt


class DataProcessor:

    def __init__(self, density=1100):
        self.DENSITY = density  # material density
        self.START = 25  # normalized start temperature

    def load_data(self, filepath, skip, colnames):
        if filepath[-1] == 'x':
            data = pd.read_excel(filepath, skiprows=skip, header=None, names=colnames)
        else:
            data = pd.read_csv(filepath, skiprows=skip, header=None, names=colnames)
        return data

    def separate_data(self, data, start, end):
        pass

    # export mass, time, temp by its collum number
    def get_mass_profile(self, data, mass_col, mass_unit=True):
        raw_mass = data.iloc[:, mass_col].values
        if mass_unit:
            raw_mass /= 100
        return raw_mass

    def get_temperature_profile(self, data, temp_col):
        return data.iloc[:, temp_col].values

    def get_time_profile(self, data, time_col, minute=True):
        raw_time = data.iloc[:, time_col].values
        if minute:
            raw_time *= 60
        return raw_time

    def export(self, data, mass_col, time_col, temp_col, minute=True, mass_unit=True):
        raw_mass = self.get_mass_profile(data, mass_col, mass_unit)
        raw_time = self.get_time_profile(data, time_col)
        raw_temp = self.get_time_profile(data, temp_col)
        return raw_mass, raw_time, raw_temp

    # Normalize the data to fixed x axis range and interval by smooth and interpolation
    def preprocess(self, x, y, smooth_rate=0.1):
        tck = interpolate.splrep(x, y, s=smooth_rate)
        end = 475
        xnew = np.linspace(self.START, end, end - self.START + 1)
        ynew = interpolate.splev(xnew, tck, der=0)
        yder = -interpolate.splev(xnew, tck, der=1)
        return xnew, ynew, yder

    # writing data into csv
    def write_data(self, norm_temp, massLossRate, filename):
        exp = np.asarray([norm_temp, massLossRate]).T
        exp = np.insert(exp, 0, [0, massLossRate[0]], axis=0)
        collum_name = ['Temp', 'MLR']
        df = pd.DataFrame(exp, columns=collum_name)
        df['MLR'][df['MLR'] < 0] = 0
        df['MLR'][df['Temp'] > 450] = 0
        plt.plot(df['Temp'], df['MLR'])
        plt.savefig(filename.replace('csv', 'png'))
        plt.show()
        df.to_csv(filename, index=False, header=collum_name, float_format='%.3f')

    # generate normalized testing data for learning
    def normalized_test_data(self, filename):
        skip_header = 1
        data = self.load_data(filename, skip_header)
        mass_col, time_col, temp_col = 2, 0, 3
        raw_mass, raw_time, raw_temp = self.export(data, mass_col, time_col, temp_col, minute=True, mass_unit=True)
        norm_temp, massLossRate = self.preprocess(raw_mass, raw_temp)
        self.write_data(norm_temp, massLossRate, filename)
        return norm_temp, massLossRate
