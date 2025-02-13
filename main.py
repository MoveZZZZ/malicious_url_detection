from DataPreprocessing import DataPreprocessing



def main():
    data_preproc = DataPreprocessing("test")

    #data_preproc.print_dataset_info(data_preproc.cleared_and_vectorized_data)
    #data_preproc.split_dataset_into_train_and_test_files(data_preproc.cleared_and_vectorized_data)
    data_preproc.print_dataset_info(data_preproc.cleared_and_vectorized_data)
    data_preproc.print_dataset_info(data_preproc.full_train)
    data_preproc.print_dataset_info(data_preproc.full_test)

    #data_preproc.refractoring_and_save_features_dataset()

    #data_preproc.clear_and_vectorize_finally_dataset(data_preproc.data_features_selected)


main()