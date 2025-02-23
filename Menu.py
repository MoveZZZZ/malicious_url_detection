import os
import pyautogui
import textwrap
from TrainModels import TrainModels
class Menu:
    def __init__(self):
        self.db_name = None
        self.log_filename = None
        self.options = {
            1: f"Training on a full dataset",
            2: f"Training on a full dataset using SMOTE",
            3: f"Training on a full dataset using ADASYN",
            4: f"Training on a full dataset using SMOTE_TOMEK",
            5: f"Training on a full dataset using SMOTE_ENN",
            6: f"Training on a full dataset using RandomUndersampling",
            91: f"Training on a full dataset with MinMaxScaler",
            92: f"Training on a full dataset using SMOTE with MinMaxScaler",
            93: f"Training on a full dataset using ADASYN with MinMaxScaler",
            94: f"Training on a full dataset using SMOTE_TOMEK with MinMaxScaler",
            95: f"Training on a full dataset using SMOTE_ENN with MinMaxScaler",
            96: f"Training on a full dataset using RandomUndersampling with MinMaxScaler",
        }
        self.datasets = [
            "custom_features",
            "bert_768",
            "bert_350",
            "bert_768_browser",
            "bert_768_browser_np"
        ]
        self.datasets_description = {
            1: "Dataset with custom-selected features",
            2: "Dataset with 768 BERT features",
            3: "Dataset with 350 BERT+PCA features",
            4: "Dataset with 768 BERT features from Selenium (includes old phishing/malware)",
            5: "Dataset with 768 BERT features from Selenium (excludes old phishing/malware and includes new phishing/malware)"
        }
        self.activation_functions = {
            1: "relu",
            2: "elu",
            3: "selu",
            4: "softmax",
            5: "sigmoid",
            6: "tanh",
            7: "gelu",
            8: "softplus"
        }
        self.optimizers = {
            1: "adamw",
            2: "sgd",
            3: "adagrad",
            4: "adadelta",
            5: "rmsprop",
            6: "nadam"
        }
        self.losses = {
            1: "categorical_crossentropy",
            2: "focal_loss",
            3: "weighted_categorical_crossentropy"
        }
        self.models_type = {
            1 : "ML",
            2: "NN"
        }
        self.models_ml = ["RFC", "lgbm", "xgb", "mlp", "log_reg"]
        self.models_nn = ["deep_mlp_3", "deep_mlp_5", "deep_mlp_7", "gnn", "AE", "RBFN"]
        self.params = {
        "option": 0,
        "model_name": "",
        "_activation_function": "",
        "_optimizer": "",
        "_loss": "",
        "_epochs": 0,
        "_num_centres_RBFL": 0,
        "_encoding_dim_AE": 0,
        "_model_params_string": ""
        }
        self.dataset = None

    def get_choice(self, prompt, valid_options):

        while True:
            try:
                choice = int(input(prompt))
                if choice in valid_options:
                    return choice
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a number.")
    def select_options(self):
        print("Select a option:")
        for key, option in self.options.items():
            print(f"{key}. {option}")
        option = self.get_choice("Enter the number of the option: ", self.options.keys())
        self.params['option'] = option
        print(f"Option selected: {self.params['_loss']}")
    def select_dataset(self):
        print("Select a dataset:")
        for key, description in self.datasets_description.items():
            print(f"{key}. {description}")
        choice = self.get_choice("Enter the number of the data set: ", self.datasets_description.keys())
        self.dataset = self.datasets[choice - 1]
        print(f"Selected dataset: {self.dataset}")

    def select_activation_function(self):
        print("Select the activation function:")
        for key, func in self.activation_functions.items():
            print(f"{key}. {func}")
        choice = self.get_choice("Enter the number of the activation function: ", self.activation_functions.keys())
        self.params['_activation_function'] = self.activation_functions[choice]
        print(f"The activation function is selected: {self.params['_activation_function']}")

    def select_optimizer(self):
        print("Select an optimizer:")
        for key, opt in self.optimizers.items():
            print(f"{key}. {opt}")
        choice = self.get_choice("Enter the optimizer number: ", self.optimizers.keys())
        self.params['_optimizer'] = self.optimizers[choice]
        print(f"An optimizer has been selected: {self.params['_optimizer']}")

    def select_loss(self):
        print("Select the loss function:")
        for key, loss in self.losses.items():
            print(f"{key}. {loss}")
        choice = self.get_choice("Enter the number of the loss function: ", self.losses.keys())
        self.params['_loss'] = self.losses[choice]
        print(f"Loss function selected: {self.params['_loss']}")

    def select_model_ml(self):
        print("Select a machine learning (ML) model:")
        for i, model in enumerate(self.models_ml, 1):
            print(f"{i}. {model}")
        valid_options = range(1, len(self.models_ml) + 1)
        choice = self.get_choice("Enter the ML model number: ", valid_options)
        self.params['model_name'] = self.models_ml[choice - 1]
        print(f"ML model selected: {self.params['model_name']}")

    def select_model_nn(self):
        print("Select a neural network (NN):")
        for i, model in enumerate(self.models_nn, 1):
            print(f"{i}. {model}")
        valid_options = range(1, len(self.models_nn) + 1)
        choice = self.get_choice("Enter the NN model number: ", valid_options)
        self.params['model_name'] = self.models_nn[choice - 1]
        print(f"The NN model has been selected: {self.params['model_name']}")
    def select_dim_for_AE(self):
        dim = self.get_choice("Enter the encoder dimension for AutoEncoder (1 to 20): ", range(1, 21))
        self.params['_encoding_dim_AE'] = dim
        print(f"The dimensionality for AE has been selected: {dim}")
    def select_centres_for_RBFL(self):
        centres = self.get_choice("Enter the number of centres for RBFL (1 to 30): ", range(1, 31))
        self.params['_num_centres_RBFL'] = centres
        print(f"The dimensionality for AE has been selected: {centres}")
    def write_model_params_string(self):
        while True:
            user_input = input("Enter a string (up to 15 characters) or press 0 to skip: ")
            if user_input == "0":
                print("Function skipped.")
                self.params["_model_params_string"] = ""
                return
            if len(user_input) <= 15:
                self.params['_model_params_string'] = user_input
                print(f"Entered string: {user_input}")
                return
            else:
                print("Error: the string must contain no more than 15 characters. Try again.")
    def write_log_file_name(self):
        while True:
            user_input = input("Enter a filename (up to 15 characters): ")
            if len(user_input) <= 15:
                self.log_filename = user_input
                print(f"Entered string: {user_input}")
                return
            else:
                print("Error: the filename must contain no more than 15 characters. Try again.")
    def select_epochs_number(self):
        while True:
            try:
                epochs = int(input("Enter the number of epochs: "))
                self.params['_epochs'] = epochs
                print(f"The number of epochs selected: {epochs}")
                break
            except ValueError:
                print("Error: enter an integer.")

    def print_summary(self):
        table_width = 90
        left_col_width = 30
        right_col_width = table_width - left_col_width - 3
        border = "=" * table_width
        print(border)
        print("|{:^{width}}|".format("SUMMARY OF SELECTED PARAMETERS", width=table_width - 2))
        print(border)
        def print_row(key, value):
            wrapped_value = textwrap.wrap(value, width=right_col_width)
            if not wrapped_value:
                wrapped_value = ['']
            print(
                "| {:<{lcol}} | {:<{rcol}} |".format(key, wrapped_value[0], lcol=left_col_width, rcol=right_col_width))
            for line in wrapped_value[1:]:
                print("| {:<{lcol}} | {:<{rcol}} |".format("", line, lcol=left_col_width, rcol=right_col_width))
        dataset_str = self.datasets_description[self.datasets.index(self.dataset) + 1]
        print_row("Dataset:", dataset_str)
        print_row("Option:", self.options.get(self.params['option'], 'N/A'))
        print_row("Model Name:", self.params.get('model_name', 'N/A'))
        print_row("Activation Function:", self.params.get('_activation_function', 'N/A'))
        print_row("Optimizer:", self.params.get('_optimizer', 'N/A'))
        print_row("Loss Function:", self.params.get('_loss', 'N/A'))
        print_row("Epochs:", str(self.params.get('_epochs', 'N/A')))
        if self.params["model_name"] == "AE":
            print_row("Encoding Dimension (AE):", str(self.params.get('_encoding_dim_AE', 'N/A')))
        elif self.params["model_name"] == "RBFL":
            print_row("Number of Centres (RBFL):", str(self.params.get('_num_centres_RBFL', 'N/A')))
        print_row("Model Params String:", self.params.get('_model_params_string', 'N/A'))
        print_row("Log_filename:", self.log_filename)
        print(border)
    def train(self):
        _TrainModel = TrainModels(self.log_filename, self.dataset)
        _TrainModel.train_model(**self.params)

    def draw_menu(self):
        print("========== Model Selection Menu ==========")
        for key, model_type in self.models_type.items():
            print(f"{key}. {model_type}")
        model_type_choice = self.get_choice("Enter the model type number: ", self.models_type.keys())
        selected_model_type = self.models_type[model_type_choice]
        print(f"Model type selected: {selected_model_type}")
        print(42*"=")
        self.select_dataset()
        print(42 * "=")
        self.select_options()
        print(42 * "=")
        if selected_model_type == "ML":
            self.select_model_ml()
            print(42 * "=")
            self.write_model_params_string()
            print(42 * "=")
        else:
            self.select_model_nn()
            print(42 * "=")
            self.select_activation_function()
            print(42 * "=")
            self.select_optimizer()
            print(42 * "=")
            self.select_loss()
            print(42 * "=")
            if self.params["model_name"] == "AE":
                self.select_dim_for_AE()
                print(42 * "=")
            elif self.params["model_name"] == "RBFL":
                self.select_centres_for_RBFL()
                print(42 * "=")
            self.select_epochs_number()
            print(42 * "=")
            self.write_model_params_string()
        print(42 * "=")
        self.write_log_file_name()
        pyautogui.hotkey('alt', 'l')
        os.system('cls')
        self.print_summary()
        self.train()



