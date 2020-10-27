import os
from data_processor import DataProcessor


def main():
    file_name = "FDS/input1.fds"
    os.system(f"fds {file_name}")

    data_processor = DataProcessor(1100)
    file_path = 'FDS/output_devc.csv'
    colnames = ["Time", "mpua", "mlrpua", "Temp"]
    raw_data = data_processor.load_data(file_path, 2, colnames)
    data = raw_data.loc[(raw_data['Temp'] > 200) & (raw_data['Temp'] < 500)]

    mass = data_processor.get_mass_profile(data, 1, False)
    temp = data_processor.get_temperature_profile(data, 3)

    norm_temp, norm_mass, mass_loss_rate = data_processor.preprocess(temp, mass, 0.1)
    data_processor.write_data(norm_temp, mass_loss_rate, file_path.replace('input', 'output'))


if __name__ == '__main__':
    main()
