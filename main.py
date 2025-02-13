from DataPreprocessing import DataPreprocessing



def main():
    data_preproc = DataPreprocessing("test")
    data_preproc.change_data_labels(data_preproc.data_full.copy())


main()