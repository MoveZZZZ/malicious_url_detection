from LogSystem import LogFileCreator
from DataPreprocessing import DataPreprocessing
from TrainTestDataPreprocessig import TrainTestDataPreprocessing
from ModelNameAndPathesCreator import ModelNameAndPathesCreator
from ConfusionMatrixAndRocCreator import CM_and_ROC_creator
import time
class TrainModels:
    def __init__(self, log_filename):
        self.LogCreator = LogFileCreator(log_filename)
        self._DataPreprocessing = DataPreprocessing(log_filename)
        self._TrainTestDataPreproc = TrainTestDataPreprocessing(log_filename)
        self._ModelNameAndPathesCreator = ModelNameAndPathesCreator(log_filename)
        self.CM_and_ROC_creator = CM_and_ROC_creator(log_filename)
        self.full_train_data = self._DataPreprocessing.full_train



    def train_model(self, option, model_name):
        self.LogCreator.print_and_write_log(f"Train {model_name} with using {self._ModelNameAndPathesCreator.define_type_of_option(option)}\n"
                                            f"{self.LogCreator.string_spit_tilds}")


        data = self.full_train_data.copy()
        X, y = self._TrainTestDataPreproc.create_X_and_Y(data)

        X_train, X_test, y_train, y_test = self._TrainTestDataPreproc.split_data_for_train_and_validation(X,y,0.2, 42)
        model, save_file_name = self._ModelNameAndPathesCreator.create_model_name_and_output_pathes(option, model_name)

        self.LogCreator.print_and_write_log(f"Start learn model {model_name}")
        model_train_time_start = time.time()
        model.fit(X_train, y_train)
        model_train_time_end = time.time()
        self.LogCreator.print_and_write_log(
            f"End learn {model_name}, time to learn: {self.LogCreator.count_time(model_train_time_start, model_train_time_end):.2f} s.\n"
            f"{self.LogCreator.string_spit_stars}")

        self.CM_and_ROC_creator.create_confusion_matrix(model, X_test, y_test, save_file_name)
        self.CM_and_ROC_creator.create_ROC(model, X_test, y_test, save_file_name)





