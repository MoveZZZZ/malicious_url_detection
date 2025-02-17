from LogSystem import LogFileCreator
from DataPreprocessing import DataPreprocessing
from TrainTestDataPreprocessing import TrainTestDataPreprocessing
from ModelNameAndPathesCreator import ModelNameAndPathesCreator
from ConfusionMatrixAndRocCreator import CM_and_ROC_creator


from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
import time
import tensorflow as tf
from CustomModels import Optimization
from transformers import BertTokenizer
import torch




class TrainModels:
    def __init__(self, log_filename):
        self.LogCreator = LogFileCreator(log_filename)
        self._DataPreprocessing = DataPreprocessing(log_filename)
        self._TrainTestDataPreproc = TrainTestDataPreprocessing(log_filename)
        self._ModelNameAndPathesCreator = ModelNameAndPathesCreator(log_filename)
        self.CM_and_ROC_creator = CM_and_ROC_creator(log_filename)
        self._Optimization = Optimization()
        print(tf.config.list_physical_devices('GPU'))
        print(torch.backends.cudnn.version())

    def data_pathes_and_model_creation(self, option, model_name, _activation_function, _optimizer, _num_centres):
        if model_name == "bert2":
            data = self._DataPreprocessing.full_train_base.copy()
            X, y = self._TrainTestDataPreproc.create_X_and_Y(data)
            tokenizer_model_name = "bert-base-uncased"
            tokenizer = BertTokenizer.from_pretrained(tokenizer_model_name)
            model, save_file_name = self._ModelNameAndPathesCreator.create_model_name_and_output_pathes(option, model_name, _activation_function, _optimizer)
            X_train, X_test, y_train, y_test = self._TrainTestDataPreproc.prepare_data_bert_2(X, y, tokenizer)
            scaler = None
        else:
            data = self._DataPreprocessing.full_train.copy()
            X, y = self._TrainTestDataPreproc.create_X_and_Y(data)
            input_size = X.shape[1]
            X_train_without_saler, X_test_without_saler, y_train, y_test = self._TrainTestDataPreproc.split_data_for_train_and_validation(X, y, 0.2,42)
            model, save_file_name = self._ModelNameAndPathesCreator.create_model_name_and_output_pathes(option, model_name, _activation_function, _optimizer, _num_centres ,input_size)
            scaler = self._TrainTestDataPreproc.create_scaler(X)
            X_train, X_test = self._TrainTestDataPreproc.scale_data(scaler, X_train_without_saler, X_test_without_saler)
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()
        return model, save_file_name, X_train, X_test, y_train, y_test, scaler

    def print_model_summary(self, model):
        print(f"Optimizer: {model.optimizer.__class__.__name__}")
        print("Loss function:", model.loss)

        # Проверяем, есть ли Sequential внутри кастомной модели
        for layer in model.layers:
            if isinstance(layer, tf.keras.Sequential):
                # Если наткнулись на Sequential, выводим его внутренние слои
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.Dropout)):
                        continue
                    print(f"Layer: {sub_layer.name}")
                    print(f"   Type: {type(sub_layer).__name__}")
                    if hasattr(sub_layer, 'activation'):
                        print(f"   Activation: {sub_layer.activation.__name__}")
                    if hasattr(sub_layer, 'kernel_initializer'):
                        print(f"   Kernel initializer: {sub_layer.kernel_initializer.__class__.__name__}")
                    if hasattr(sub_layer, 'units'):
                        print(f"   Units: {sub_layer.units}")
            else:
                # Обычные слои
                if isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.Dropout)):
                    continue
                print(f"Layer: {layer.name}")
                print(f"   Type: {type(layer).__name__}")
                if hasattr(layer, 'activation'):
                    print(f"   Activation: {layer.activation.__name__}")
                if hasattr(layer, 'kernel_initializer'):
                    print(f"   Kernel initializer: {layer.kernel_initializer.__class__.__name__}")
                if hasattr(layer, 'units'):
                    print(f"   Units: {layer.units}")

        # Вывод стандартного summary
        print(model.model.summary())
    def train_model(self, option, model_name, _activation_function, _optimizer, _epochs = 1, _num_centres=10):
        self.LogCreator.print_and_write_log(f"Train {model_name} with using {self._ModelNameAndPathesCreator.define_type_of_option(option)}\n"
                                            f"{self.LogCreator.string_spit_tilds}")

        model, save_file_name, X_train, X_test, y_train, y_test, scaler = self.data_pathes_and_model_creation(option, model_name, _activation_function, _optimizer, _num_centres)

        self.LogCreator.print_and_write_log(f"Start learn model {model_name}")
        model_train_time_start = time.time()

        if isinstance(model, models.Model):
            if y_train.ndim == 1:
                y_train = to_categorical(y_train, num_classes=4)
                y_test = to_categorical(y_test, num_classes=4)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
            #'categorical_crossentropy'
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            model.compile(optimizer=_optimizer, loss = self._Optimization.focal_loss(), metrics = ['accuracy'])
            self.print_model_summary(model)
            history = model.fit(X_train, y_train, epochs=_epochs, batch_size=32, validation_data=(X_test, y_test),
                      verbose=1, callbacks=[early_stopping])
            self.CM_and_ROC_creator.create_confusion_matrix(model, X_test, y_test, save_file_name)
            self.CM_and_ROC_creator.create_ROC(model, X_test, y_test, save_file_name)
            self.CM_and_ROC_creator.create_plot_traning_history(model_name, history, save_file_name)
            self.check_test_data_custom(model, scaler, save_file_name)
        elif model_name == "tabnet":
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric=["accuracy"],
                batch_size=256,
                max_epochs = _epochs,
                patience=5,
            )
            self.CM_and_ROC_creator.create_confusion_matrix(model, X_test, y_test, save_file_name)
            self.CM_and_ROC_creator.create_ROC(model, X_test, y_test, save_file_name)
        elif model_name == "bert2":
            batch_size = 16
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            model.compile(optimizer="adamw", loss=loss_fn, metrics=["accuracy"])
            history = model.fit(
                [X_train["input_ids"], X_train["attention_mask"]], y_train,
                validation_data=([X_test["input_ids"], X_test["attention_mask"]], y_test),
                batch_size=batch_size, epochs=_epochs)
            self.CM_and_ROC_creator.create_confusion_matrix(model, X_test, y_test, save_file_name)
            self.CM_and_ROC_creator.create_ROC(model, X_test, y_test, save_file_name)
            self.CM_and_ROC_creator.create_plot_traning_history(model_name, history, save_file_name)
        else:
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            model.fit(X_train, y_train)
            self.CM_and_ROC_creator.create_confusion_matrix(model, X_test, y_test, save_file_name)
            self.CM_and_ROC_creator.create_ROC(model, X_test, y_test, save_file_name)

        model_train_time_end = time.time()
        self.LogCreator.print_and_write_log(
            f"End learn {model_name}, time to learn: {self.LogCreator.count_time(model_train_time_start, model_train_time_end):.2f} s.\n"
            f"{self.LogCreator.string_spit_stars}")



    def check_test_data_custom(self, model, scaler, save_filename):

        data = self._DataPreprocessing.full_test.copy()
        X, y_test = self._TrainTestDataPreproc.create_X_and_Y(data)
        X_test_scaled, _ = self._TrainTestDataPreproc.scale_data(scaler, X)

        if y_test.ndim == 1 or y_test.shape[1] == 1:
            y_test = to_categorical(y_test, num_classes=4)
        score = model.evaluate(X_test_scaled, y_test, verbose=0)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        self.CM_and_ROC_creator.create_confusion_matrix(model, X_test_scaled, y_test, save_filename+"_test_data")




