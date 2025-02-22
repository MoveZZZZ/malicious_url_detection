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
    # loss = (
    #     "categorical_crossentropy",
    #     "focal_loss",
    #     "weighted_categorical_crossentropy"
    # )
    # dataset = (
    #     "custom_features" -> dataset with custom selected features
    #     "bert_768" -> dataset with 768 selected bert features
    #     "bert_350" -> dataset with 350 selected bert+PCA features
    # )
    #option, model_name, _activation_function, _optimizer, _loss, _epochs , _num_centres_RBFL, _encoding_dim_AE, _model_params_string=""

    dataset = "bert_768"
    log_filename = "test"
    loss = "weighted_categorical_crossentropy"

    _TrainModels = TrainModels(log_filename, dataset)
    _dp = DataPreprocessing(log_filename)
    _dp.select_bert_features()

    #_dp.split_large_csv_into_train_and_test()

    #_dp.extract_features_bert(["mp3raid.com/music/krizz_kaliko.html"])
    #_TrainModels.train_model(911, "AE", "relu", "adamw","categorical_crossentropy",30, 1,8,"")
    # _TrainModels.train_model(91, "RFC","", "","", 1, 1, 1, "bert_350")
    # _TrainModels.train_model(911, "RFC","", "","", 1, 1, 1, "bert_350")
    #
    # _TrainModels.train_model(91, "lgbm","", "","", 1, 1, 1, "bert_350")
    # _TrainModels.train_model(911, "lgbm","", "","", 1, 1, 1, "bert_350")
    #
    # _TrainModels.train_model(91, "xgb","", "","", 1, 1, 1, "bert_350")
    # _TrainModels.train_model(911, "xgb","", "","", 1, 1, 1, "bert_350")










main()