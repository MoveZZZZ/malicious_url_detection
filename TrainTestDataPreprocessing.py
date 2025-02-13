from LogSystem import LogFileCreator
import time
from sklearn.model_selection import train_test_split
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