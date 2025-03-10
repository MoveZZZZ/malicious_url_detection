import os
import pyautogui
import textwrap
from TrainModels import TrainModels
from LogSystem import LogFileCreator
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
        self.models_nn = ["deep_mlp_3", "deep_mlp_5", "deep_mlp_7", "gnn", "AE", "RBFL"]
        self.params = {
        "option": 0,
        "model_name": "",
        "_activation_function": "",
        "_optimizer": "",
        "_loss": "",
        "_epochs": 0,
        "_num_centres_RBFL": 0,
        "use_kmeans": False,
        "_encoding_dim_AE": 0,
        "_model_params_string": None
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
        dim = self.get_choice("Enter the encoder dimension for AutoEncoder (1 to 500): ", range(1, 501))
        self.params['_encoding_dim_AE'] = dim
        print(f"The dimensionality for AE has been selected: {dim}")
    def select_centres_for_RBFL(self):
        centres = self.get_choice("Enter the number of centres for RBFL (1 to 10000): ", range(1, 10001))
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
            user_input = input("Enter a filename (up to 30 characters): ")
            if len(user_input) <= 30:
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
        print()
        table_width = 95
        left_col_width = 25
        right_col_width = table_width - left_col_width - 8
        border = "=" * (table_width-1)
        _log = LogFileCreator(self.log_filename)
        summary_lines = []
        summary_lines.append(border)
        summary_lines.append("|{:^{width}}|".format("SUMMARY OF SELECTED PARAMETERS", width=table_width-3))
        summary_lines.append(border)

        def add_row(key, value):
            wrapped_value = textwrap.wrap(value, width=right_col_width)
            if not wrapped_value:
                wrapped_value = ['']
            summary_lines.append("| {:<{lcol}} | {:<{rcol}} |".format(key, wrapped_value[0],
                                                                      lcol=left_col_width, rcol=right_col_width))
            for line in wrapped_value[1:]:
                summary_lines.append("| {:<{lcol}} | {:<{rcol}} |".format("", line,
                                                                          lcol=left_col_width, rcol=right_col_width))

        dataset_str = self.datasets_description[self.datasets.index(self.dataset) + 1]
        add_row("Dataset:", dataset_str)
        add_row("Option:", self.options.get(self.params['option'], 'N/A'))
        add_row("Model Name:", self.params.get('model_name', 'N/A'))
        if self.params["model_name"] in self.models_nn and self.params["model_name"] != "RBFL":
            add_row("Activation Function:", self.params.get('_activation_function', 'N/A'))
        elif self.params["model_name"] == "RBFL":
            add_row("Number of Centres (RBFL):", str(self.params.get('_num_centres_RBFL', 'N/A')))
            add_row("Use kmeans (RBFL):", "Yes" if self.params["use_kmeans"] == True else "No")
        if self.params["model_name"] in self.models_nn:
            add_row("Optimizer:", self.params.get('_optimizer', 'N/A'))
            add_row("Loss Function:", self.params.get('_loss', 'N/A'))
            add_row("Epochs:", str(self.params.get('_epochs', 'N/A')))
        if self.params["model_name"] == "AE":
            add_row("Encoding Dimension (AE):", str(self.params.get('_encoding_dim_AE', 'N/A')))
        add_row("Log_filename:", self.log_filename)
        add_row("Model Params String:",
                "N/A" if self.params.get('_model_params_string', 'N/A') == "" else self.params.get(
                    '_model_params_string', 'N/A'))
        summary_lines.append(border)
        summary_text = "\n".join(summary_lines)
        _log.print_and_write_log(summary_text)
    def use_kmeans_option(self):
        while True:
            try:
                print("Use kmeans for centroids?:\n1: Yes\n0: No")
                use_kmeans = int(input("Enter the kmeans option: "))
                if use_kmeans not in [0,1]:
                    print("Error: enter an 0 or 1. Try again.")
                else:
                    self.params["use_kmeans"] = True if use_kmeans == 1 else False
                    return
            except ValueError:
                print("Error: enter an integer.")


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
        else:
            self.select_model_nn()
            print(42 * "=")
            self.select_optimizer()
            print(42 * "=")
            self.select_loss()
            print(42 * "=")
            if self.params["model_name"] == "RBFL":
                self.select_centres_for_RBFL()
                print(42 * "=")
                self.use_kmeans_option()
                print(42 * "=")
            else:
                self.select_activation_function()
                print(42 * "=")
                if self.params["model_name"] == "AE":
                    self.select_dim_for_AE()
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



