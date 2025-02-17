from LogSystem import LogFileCreator
import torch
from CustomModels import DeepMLP_3, DeepMLP_5, GNN
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from transformers import TFBertForSequenceClassification
class ModelNameAndPathesCreator:
    def __init__(self, log_filename):
        self.LogCreator = LogFileCreator(log_filename)

    def create_model_name_and_output_pathes(self, option, model_name, input_size=None, num_classes=4):
        model = None
        end_file_name = self.define_type_of_option(option)

        if model_name == "RFC":
            model = RandomForestClassifier()
            save_file_name = f"RandomForestClassifier_{end_file_name}"
        elif model_name == "lgbm":
            model = LGBMClassifier(
                objective="multiclass",
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
                hidden_layer_sizes=(128, 64, 32),
                random_state=42,
                max_iter=200,
            )
            save_file_name = f"MLPClassifier_{end_file_name}"
        elif model_name == "tabnet":
            model = TabNetClassifier(
                optimizer_params=dict(lr=2e-2),
                verbose = 1,
                scheduler_params={"step_size": 10, "gamma": 0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                seed=42

            )
            save_file_name = f"TabNetClassifier_{end_file_name}"
        elif model_name == "deep_mlp_3":
            if input_size is None:
                raise ValueError("DeepMLP requires input_size to be specified")
            model = DeepMLP_3(input_size, num_classes)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            save_file_name = f"DeepMLP_3_layer_{end_file_name}"
        elif model_name == "deep_mlp_5":
            if input_size is None:
                raise ValueError("DeepMLP requires input_size to be specified")
            model = DeepMLP_5(input_size, num_classes)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            save_file_name = f"DeepMLP_5_layer_{end_file_name}"
        elif model_name == "gnn":
            if input_size is None:
                raise ValueError("GNN requires input_size to be specified")
            model = GNN(input_size, 128, num_classes)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            save_file_name = f"GNN_{end_file_name}"
        elif model_name == "bert2":
            model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
            save_file_name = f"BERT_{end_file_name}"
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