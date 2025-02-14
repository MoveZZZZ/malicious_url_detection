from LogSystem import LogFileCreator
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
class TrainTestDataPreprocessing:
    def __init__(self, log_filename):
        self.LogCreator = LogFileCreator(log_filename)


    def create_X_and_Y(self, data):
        X = data.drop(['type'], axis = 1)
        y = data['type']
        return X,y

    def split_data_for_train_and_validation(self, X, y, _test_size, _random_state):
        self.LogCreator.print_and_write_log("Start split data on train and test")
        spit_time_start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=_test_size, random_state=_random_state)
        spit_time_end = time.time()
        self.LogCreator.print_and_write_log(f"End split data on train and test. "
                                            f"Time to split: {self.LogCreator.count_time(spit_time_start, spit_time_end):.2f} s.\n"
                                            f"{self.LogCreator.string_spit_stars}")
        return X_train, X_test, y_train, y_test

    def create_scaler(self, X):
        self.LogCreator.print_and_write_log("Start creating scaler")
        time_to_create_scaler_start = time.time()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X)

        time_to_create_scaler_end = time.time()
        self.LogCreator.print_and_write_log(
            f"End creating scaler. Time to create scaler: "
            f"{self.LogCreator.count_time(time_to_create_scaler_start, time_to_create_scaler_end):.2f} s.\n"
            f"{self.LogCreator.string_spit_stars}"
        )
        return scaler
    def scale_data(self, scaler,X_train, X_test=None):
        self.LogCreator.print_and_write_log("Start scaling data")
        time_to_scale_start = time.time()

        X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
        if X_test is not None:
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        else:
            X_test_scaled = None
        time_to_scale_end = time.time()
        self.LogCreator.print_and_write_log(
            f"End scaling data. Time to scale data: "
            f"{self.LogCreator.count_time(time_to_scale_start, time_to_scale_end):.2f} s.\n"
            f"{self.LogCreator.string_spit_stars}"
        )
        return X_train_scaled, X_test_scaled