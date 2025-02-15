import keras
from LogSystem import LogFileCreator
from DataPreprocessing import DataPreprocessing
from TrainTestDataPreprocessing import TrainTestDataPreprocessing
from ModelNameAndPathesCreator import ModelNameAndPathesCreator
from ConfusionMatrixAndRocCreator import CM_and_ROC_creator
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import time
import tensorflow as tf
from transformers import BertTokenizer
import torch
from tqdm import tqdm


class TrainModels:
    def __init__(self, log_filename):
        self.LogCreator = LogFileCreator(log_filename)
        self._DataPreprocessing = DataPreprocessing(log_filename)
        self._TrainTestDataPreproc = TrainTestDataPreprocessing(log_filename)
        self._ModelNameAndPathesCreator = ModelNameAndPathesCreator(log_filename)
        self.CM_and_ROC_creator = CM_and_ROC_creator(log_filename)
        print(tf.config.list_physical_devices('GPU'))
        print(torch.backends.cudnn.version())


    def train_model(self, option, model_name):
        self.LogCreator.print_and_write_log(f"Train {model_name} with using {self._ModelNameAndPathesCreator.define_type_of_option(option)}\n"
                                            f"{self.LogCreator.string_spit_tilds}")



        data = self._DataPreprocessing.full_train.copy()
        X, y = self._TrainTestDataPreproc.create_X_and_Y(data)
        input_size = X.shape[1]


        X_train, X_test, y_train, y_test = self._TrainTestDataPreproc.split_data_for_train_and_validation(X,y,0.2, 42)
        model, save_file_name = self._ModelNameAndPathesCreator.create_model_name_and_output_pathes(option, model_name, input_size)


        # USE SCALAR
        scaler = self._TrainTestDataPreproc.create_scaler(X)
        X_train_scaled, X_test_scaled = self._TrainTestDataPreproc.scale_data(scaler, X_train, X_test)

        X_train_scaled = X_train_scaled.to_numpy()
        X_test_scaled = X_test_scaled.to_numpy()


        self.LogCreator.print_and_write_log(f"Start learn model {model_name}")
        model_train_time_start = time.time()

        if isinstance(model, models.Model):
            if y_train.ndim == 1:
                y_train = to_categorical(y_train, num_classes=4)
                y_test = to_categorical(y_test, num_classes=4)
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
            model.compile(optimizer=keras.optimizers.Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])
            history = model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, validation_data=(X_test_scaled, y_test),
                      verbose=1, callbacks=[early_stopping])
            self.CM_and_ROC_creator.create_confusion_matrix_for_custom_models(model, X_test_scaled, y_test, save_file_name)
            self.CM_and_ROC_creator.create_ROC_custom(model, X_test_scaled, y_test, save_file_name)
            self.CM_and_ROC_creator.create_plot_traning_history(model_name, history, save_file_name)
            self.check_test_data_custom(model, scaler, save_file_name)

        elif model_name == "tabnet":
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                eval_metric=["accuracy"],
                batch_size=256,
                max_epochs = 1,
                patience=5,
            )
            self.CM_and_ROC_creator.create_confusion_matrix(model, X_test_scaled, y_test, save_file_name)
            self.CM_and_ROC_creator.create_ROC(model, X_test_scaled, y_test, save_file_name)

        else:
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            model.fit(X_train_scaled, y_train)
            self.CM_and_ROC_creator.create_confusion_matrix(model, X_test_scaled, y_test, save_file_name)
            self.CM_and_ROC_creator.create_ROC(model, X_test_scaled, y_test, save_file_name)

        model_train_time_end = time.time()
        self.LogCreator.print_and_write_log(
            f"End learn {model_name}, time to learn: {self.LogCreator.count_time(model_train_time_start, model_train_time_end):.2f} s.\n"
            f"{self.LogCreator.string_spit_stars}")

    def train_bert_based_model(self, option, model_name):
        self.LogCreator.print_and_write_log(
            f"Train {model_name} with using {self._ModelNameAndPathesCreator.define_type_of_option(option)}\n"
            f"{self.LogCreator.string_spit_tilds}")

        data = self._DataPreprocessing.full_train_base.copy()
        X, y = self._TrainTestDataPreproc.create_X_and_Y(data)

        self.LogCreator.print_and_write_log(f"Start tokenize data {model_name}")
        tokenize_start = time.time()

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        batch_size = 10_000
        X_tokenized = {"input_ids": [], "attention_mask": []}

        for i in tqdm(range(0, len(X), batch_size), desc="Tokenizing", unit="batch"):
            batch = X["url"][i:i + batch_size].tolist()
            tokenized_batch = tokenizer(batch, padding=True, truncation=True, max_length=128,
                                        return_tensors="tf")
            X_tokenized["input_ids"].append(tokenized_batch["input_ids"])
            X_tokenized["attention_mask"].append(tokenized_batch["attention_mask"])

        # Проверьте размеры
        for i, input_ids_batch in enumerate(X_tokenized["input_ids"]):
            print(f"Batch {i} shape: {input_ids_batch.shape}")

        X_tokenized["input_ids"] = tf.concat(X_tokenized["input_ids"], axis=0)
        X_tokenized["attention_mask"] = tf.concat(X_tokenized["attention_mask"], axis=0)

        tokenize_end = time.time()
        self.LogCreator.print_and_write_log(
            f"End tokenize data, time to tokenize: {self.LogCreator.count_time(tokenize_start, tokenize_end):.2f} s.\n"
            f"{self.LogCreator.string_spit_stars}")

        X_input_ids = X_tokenized["input_ids"]
        X_attention_mask = X_tokenized["attention_mask"]
        assert X_input_ids.shape[0] == len(y), f"The sizes don't match: X_input_ids={X_input_ids.shape[0]}, y={len(y)}"
        assert X_attention_mask.shape[0] == len(
            y), f"The sizes don't match: X_attention_mask={X_attention_mask.shape[0]}, y={len(y)}"


        X_train, X_test, y_train, y_test = self._TrainTestDataPreproc.split_data_for_train_and_validation_bert(X_input_ids, X_attention_mask,
                                                                                         y, 0.2, 42)

        model, save_file_name = self._ModelNameAndPathesCreator.create_model_name_and_output_pathes(option, model_name)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.LogCreator.print_and_write_log(f"Start learning model {model_name}")
        model_train_time_start = time.time()
        model.fit([X_train["input_ids"], X_train["attention_mask"]], y_train,
                  validation_data=([X_test["input_ids"], X_test["attention_mask"]], y_test), epochs=5, batch_size=8)
        model_train_time_end = time.time()
        self.LogCreator.print_and_write_log(
            f"End learn {model_name}, time to learn: {self.LogCreator.count_time(model_train_time_start, model_train_time_end):.2f} s.\n"
            f"{self.LogCreator.string_spit_stars}")
        self.CM_and_ROC_creator.create_confusion_matrix_for_custom_models(model, X_test, y_test, save_file_name)
        self.CM_and_ROC_creator.create_ROC_custom(model, X_test, y_test, save_file_name)
        self.CM_and_ROC_creator.create_plot_traning_history(model_name, X_test, save_file_name)


    def check_test_data_custom(self, model, scaler, save_filename):

        data = self._DataPreprocessing.full_test.copy()
        X, y_test = self._TrainTestDataPreproc.create_X_and_Y(data)
        X_test_scaled, _ = self._TrainTestDataPreproc.scale_data(scaler, X)

        if y_test.ndim == 1 or y_test.shape[1] == 1:
            y_test = to_categorical(y_test, num_classes=4)
        score = model.evaluate(X_test_scaled, y_test, verbose=0)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        self.CM_and_ROC_creator.create_confusion_matrix_for_custom_models(model, X_test_scaled, y_test, save_filename+"_test_data")




