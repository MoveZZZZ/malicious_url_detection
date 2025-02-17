from TrainModels import TrainModels
from DataPreprocessing import DataPreprocessing

def main():
    #RFC, lgbm, xgb, mlp, tabnet, deep_mlp_3, deep_mlp_5, gnn, rbfn
    #option, model_name, _activation_function, _optimizer, _epochs = 1, _num_centres=10
    log_filename = "tran_first"
    _TrainModels = TrainModels(log_filename)
    #_TrainModels.train_bert_based_model(1,"bert")
    _TrainModels.train_model(1, "rbfn", '', 'adam',30, 400)
    #_TrainModels.train_model(1,"deep_mlp")







main()