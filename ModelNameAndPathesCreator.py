from LogSystem import LogFileCreator
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

class ModelNameAndPathesCreator:
    def __init__(self, log_filename):
        self.LogCreator = LogFileCreator(log_filename)

    def create_model_name_and_output_pathes(self, option, model_name):
        model = None
        end_file_name = self.define_type_of_option(option)

        if model_name == "RFC":
            model = RandomForestClassifier()
            save_file_name = f"RandomForestClassifier_{end_file_name}"
        elif model_name == "lgbm":
            model = LGBMClassifier(
                objective="multilass",
                verbose=-1,
                force_col_wise=True,
            )
            save_file_name = f"LGBMClassifier_{end_file_name}"
        elif model_name == "xgb":
            model = XGBClassifier(eval_metric="mlogloss")
            save_file_name = f"XGBClassifier_{end_file_name}"
        elif model_name == "mlp":
            model = MLPClassifier(
                solver="adam",
                alpha=1e-4,
                hidden_layer_sizes=(100, 50),
                random_state=42,
                max_iter=200,
            )
            save_file_name = f"MLPClassifier_{end_file_name}"
        else:
            print("Model is not defined!")
            return None, None

        return model, save_file_name


    def define_type_of_option(self, option):
        file_end_map = {
            1: "full_data",
            2: "smote_oversampling"
        }
        return file_end_map.get(option, "UNKNOWN_OPTION")