class Menu:
    def __init__(self):
        self.db_name = None
        self.log_filename = None
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
        self.models_ml = ["RFC", "lgbm", "xgb", "mlp", "log_reg"]
        self.models_nn = ["deep_mlp_3", "deep_mlp_5", "gnn"]
        self.models_ae = ["AE"]
        self.models_rbfn = ["RBFN"]
        self.params = {}




