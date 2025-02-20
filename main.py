from TrainModels import TrainModels
from DataPreprocessing import DataPreprocessing

def main():
    #RFC, lgbm, xgb, mlp, tabnet, log_reg, deep_mlp_3, deep_mlp_5, gnn, rbfn
    # options = (
    #     1: "full_data",
    #     2: "smote",
    #     3: "adasyn",
    #     91: "bert_features_full_data",
    #     92: "bert_features_smote",
    #     93: "bert_features_adasyn",
    #     911: "bert_features_full_data_with_minmax_scaler",
    #     912: "bert_features_smote_with_minmax_scaler",
    #     913: "bert_features_adasyn_with_minmax_scaler"
    # )
    #option, model_name, _activation_function, _optimizer, _epochs = 1, _num_centres=10
    log_filename = "AE_test_relu_selu_adam_sgd"
    _TrainModels = TrainModels(log_filename)

    #_Data_Preproc = DataPreprocessing(log_filename)
    #_Data_Preproc.select_bert_features()
    #_Data_Preproc.split_large_csv_into_train_and_test()
    #_TrainModels.train_model(912,"xgb", "relu", "adam", 30)
    _TrainModels.train_model(911, "deep_mlp_5", 'relu', 'adam', 30, 400, 8, "512-128_stable_bert_350")
    #_Data_Preproc.select_bert_features_with_PCA()
    # #_TrainModels.train_bert_based_model(1,"bert")
    #
    # _TrainModels.train_model(99, "deep_mlp_3", 'relu', 'adam',30, 400, 10, "512-128_cleared_2")
    #
    # # _TrainModels.train_model(2, "deep_mlp_3", 'relu', 'adam',30, 400, 10, "512-128")
    # # _TrainModels.train_model(3, "deep_mlp_3", 'relu', 'adam',30, 400, 10, "512-128")
    # #
    # #
    # # _TrainModels.train_model(1, "deep_mlp_3", 'selu', 'adam',30, 400, 10, "512-128")
    # # _TrainModels.train_model(2, "deep_mlp_3", 'selu', 'adam',30, 400, 10, "512-128")
    # # _TrainModels.train_model(3, "deep_mlp_3", 'selu', 'adam',30, 400, 10, "512-128")
    # #
    # # _TrainModels.train_model(1, "deep_mlp_3", 'relu', 'SGD',30, 400, 10, "512-128")
    # # _TrainModels.train_model(2, "deep_mlp_3", 'relu', 'SGD',30, 400, 10, "512-128")
    # # _TrainModels.train_model(3, "deep_mlp_3", 'relu', 'SGD',30, 400, 10, "512-128")
    # #
    # #
    # # _TrainModels.train_model(1, "deep_mlp_3", 'selu', 'SGD',30, 400, 10, "512-128")
    # # _TrainModels.train_model(2, "deep_mlp_3", 'selu', 'SGD',30, 400, 10, "512-128")
    # # _TrainModels.train_model(3, "deep_mlp_3", 'selu', 'SGD',30, 400, 10, "512-128")
    #
    # #_TrainModels.train_model(1, "deep_mlp_5", 'sigmoid', 'adam',2, 400, 18)








main()