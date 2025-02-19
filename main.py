from TrainModels import TrainModels
from DataPreprocessing import DataPreprocessing

def main():
    #RFC, lgbm, xgb, mlp, tabnet, deep_mlp_3, deep_mlp_5, gnn, rbfn
    #option, model_name, _activation_function, _optimizer, _epochs = 1, _num_centres=10
    log_filename = "AE_test_relu_selu_adam_sgd"
    _TrainModels = TrainModels(log_filename)

    _Data_Preproc = DataPreprocessing(log_filename)

    _Data_Preproc.select_bert_features()

    # #_Data_Preproc.refractoring_and_save_features_dataset()
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