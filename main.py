from DataPreprocessing import DataPreprocessing



def main():
    data_preproc = DataPreprocessing("test")
    data_preproc.change_data_labels(data_preproc.read_data(data_preproc.base_dataset_path).copy())
    data_preproc.refractoring_and_save_features_dataset()

main()