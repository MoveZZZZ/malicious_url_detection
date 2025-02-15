from TrainModels import TrainModels
from DataPreprocessing import DataPreprocessing

def main():
    #RFC, lgbm, xgb, mlp, tabnet, deep_mlp, gnn
    log_filename = "tran_first"
    _TrainModels = TrainModels(log_filename)
    #_TrainModels.train_bert_based_model(1,"bert")
    _TrainModels.train_model(1,"deep_mlp")







main()