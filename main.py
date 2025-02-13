from TrainModels import TrainModels


def main():
    log_filename = "tran_first"
    _TrainModels = TrainModels(log_filename)
    _TrainModels.train_model(1,"xgb")






main()