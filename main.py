from TrainModels import TrainModels
from DataPreprocessing import DataPreprocessing

def main():
    #RFC, lgbm, xgb, mlp, tabnet, log_reg, deep_mlp_3, deep_mlp_5, gnn, rbfn
    # options = (
    #     1: f"{db_name}_features_full_data",
    #     2: f"{db_name}_features_smote_oversampling",
    #     3: f"{db_name}_features_adasyn_oversampling",
    #     4: f"{db_name}_features_smote_tomek",
    #     5: f"{db_name}_features_smote_enn",
    #     6: f"{db_name}_features_random_under",
    #     91: f"{db_name}_features_full_data_scaled",
    #     92: f"{db_name}_features_smote_oversampling_scaled",
    #     93: f"{db_name}_features_adasyn_oversampling_scaled",
    #     94: f"{db_name}_features_smote_tomek_scaled",
    #     95: f"{db_name}_features_smote_enn_scaled",
    #     96: f"{db_name}_features_random_under_scaled",
    # )
    # loss = (
    #     "categorical_crossentropy",
    #     "focal_loss",
    #     "weighted_categorical_crossentropy"
    # )
    # dataset = (
    #     "custom_features" -> dataset with custom selected features
    #     "bert_768" -> dataset with 768 selected bert features
    #     "bert_350" -> dataset with 350 selected bert+PCA features
    #     "bert_768_browser" -> dataset with 768 selected bert features after seleniunm + new phishing/malware (with old)
    #     "bert_768_browser_np" -> dataset with 768 selected bert features after seleniunm + new phishing/malware (without old)
    # )
    #option, model_name, _activation_function, _optimizer, _loss, _epochs , _num_centres_RBFL, _encoding_dim_AE, _model_params_string=""

    log_filename= "test"
    dataset = "bert_768_browser_np"

    NN_PARAMS ={
        "option": 93,
        "model_name": "deep_mlp_3",
        "_activation_function": "relu",
        "_optimizer": "adamw",
        "_loss": "categorical_crossentropy",
        "_epochs": 30,
        "_num_centres_RBFL": 10,
        "_encoding_dim_AE": 8,
        "_model_params_string": "512-64_layer"
    }
    ML_PARAMS ={
        "option": 96,
        "model_name": "xgb",
        "_activation_function": "",
        "_optimizer": "",
        "_loss": "",
        "_epochs": 0,
        "_num_centres_RBFL": 0,
        "_encoding_dim_AE": 0,
        "_model_params_string": ""
    }

    _TrainModels = TrainModels(log_filename, dataset)
    _TrainModels.train_model(**NN_PARAMS)









main()