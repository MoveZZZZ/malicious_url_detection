from TrainModels import TrainModels


def main():
    #RFC, lgbm, xgb, mlp, tabnet, deep_mlp, gnn
    log_filename = "tran_first"
    _TrainModels = TrainModels(log_filename)
    _TrainModels.train_model(1,"tabnet")






main()