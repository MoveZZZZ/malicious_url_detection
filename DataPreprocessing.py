import pandas as pd
import numpy as np
import time, warnings
from sklearn.model_selection import StratifiedShuffleSplit
from LogSystem import LogFileCreator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from imblearn.under_sampling import TomekLinks, ClusterCentroids, NearMiss
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTETomek


class DataPreprocessing:
    def __init__(self, log_filename):
        self.cleared_data_file_path_base = "D:/PWR/Praca magisterska/Dataset/dataset_mall.csv"
        #self.data_for_train_model = "C:/Users/maksf/Desktop/MachineLearningCSV_F/MachineLearningCVE/for_train_90_perc.csv"
        #self.data_for_test_model = "C:/Users/maksf/Desktop/MachineLearningCSV_F/MachineLearningCVE/for_full_test_10_perc.csv"
        self.data_full = self.read_data(self.cleared_data_file_path_base)
        #self.data_train = self.read_data(self.data_for_train_model)
        #self.data_test = self.read_data(self.data_for_test_model)
        self.LogCreator = LogFileCreator(log_filename)
        self.label_mapping_url = {
            'benign': 0,
            'defacement': 1,
            'phishing': 2,
            'malware': 3
        }
    def read_data(self, file_path):
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            print(f"Failed to read file {file_path}: {e}")
            return None

    def print_dataset_info (self, data):
        print(data['type'].value_counts())

    def print_class_count(self, data):
        print(f"BENIGN: {(data['type'] == 'benign').sum()}")
        print(f"DEFACEMENT: {(data['type'] != 'defacement').sum()}")
        print(f"PHISHING: {(data['type'] != 'phishing').sum()}")
        print(f"MALWARE: {(data['type'] != 'malware').sum()}")

    def change_data_labels(self, _data):
        self.print_dataset_info(_data)
        self.LogCreator.print_and_write_log("Start change data labels")
        change_data_labels_start = time.time()
        data = _data.copy()
        data['type'] = data['type'].map(self.label_mapping_url)
        change_data_labels_end = time.time()
        self.LogCreator.print_and_write_log(
            f"End change data labels. Time to change: {self.LogCreator.count_time(change_data_labels_start, change_data_labels_end):.2f} s.\n"
            f"{self.LogCreator.string_spit_stars}")
        self.print_dataset_info(data)
        return data



    def calculate_url_lenght(self, url):
        return len(url)
