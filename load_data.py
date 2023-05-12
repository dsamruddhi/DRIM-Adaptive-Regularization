import os
from scipy.io import loadmat

from config import *


class LoadData:

    real_path = r"C:\Users\dsamr\OneDrive - HKUST Connect\MPHIL RESEARCH\DATA\DRIM_ADAPTIVE_3\scatterer_data_inverse_20"
    total_power_path = r"C:\Users\dsamr\OneDrive - HKUST Connect\MPHIL RESEARCH\DATA\DRIM_ADAPTIVE_3\measurement_scatterer"

    num_samples = 10
    test_size = 0.1

    @staticmethod
    def get_files(filepath):
        files = os.listdir(filepath)
        files.sort(key=lambda x: int(x.strip(".mat")))
        return files

    @staticmethod
    def get_ground_truth_data():
        scatterers = []
        files = LoadData.get_files(LoadData.real_path)
        num_files = LoadData.num_samples if LoadData.num_samples <= len(files) else len(files)
        for file in files[:num_files]:
            filename = os.path.join(LoadData.real_path, file)
            scatterer = loadmat(filename)["scatterer"]
            scatterers.append(np.real(scatterer))
        return scatterers

    @staticmethod
    def get_total_power():
        powers = []
        files = LoadData.get_files(LoadData.total_power_path)
        num_files = LoadData.num_samples if LoadData.num_samples <= len(files) else len(files)
        for file in files[:num_files]:
            filename = os.path.join(LoadData.total_power_path, file)
            total_power = loadmat(filename)["total_power"]
            powers.append(np.real(total_power))
        return powers

    @staticmethod
    def check_data_sanctity(arrays):
        for array in arrays:
            assert not np.isnan(array).any()

    @staticmethod
    def split_data(array, test_size):
        test_data_len = int(len(array) * test_size)
        train_data_len = len(array) - test_data_len
        train_data, test_data = array[:train_data_len, :], array[train_data_len:, :]
        return train_data, test_data

    @staticmethod
    def main():

        real_data = LoadData.get_ground_truth_data()
        real_data = np.asarray(real_data)

        powers = LoadData.get_total_power()
        powers = np.asarray(powers)

        LoadData.check_data_sanctity([real_data, powers])
        real_data_train, real_data_test = LoadData.split_data(real_data, LoadData.test_size)
        powers_train, powers_test = LoadData.split_data(powers, LoadData.test_size)

        print(f"real train: {real_data_train.shape}, real test: {real_data_test.shape}",
              f"powers train: {powers_train.shape}, powers test: {powers_test.shape}")

        return real_data_train, real_data_test, powers_train, powers_test
