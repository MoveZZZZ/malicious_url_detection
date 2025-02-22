import numpy as np

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
import joblib
from sklearn.metrics import accuracy_score, classification_report

class TrainModels:
    def __init__(self, log_filename, _data_option):
        self.LogCreator = LogFileCreator(log_filename)
        self._DataPreprocessing = DataPreprocessing(log_filename)
        self._TrainTestDataPreproc = TrainTestDataPreprocessing(log_filename)
        self._ModelNameAndPathesCreator = ModelNameAndPathesCreator(log_filename)
        self.CM_and_ROC_creator = CM_and_ROC_creator(log_filename)
        self._Optimization = Optimization()
        print(tf.config.list_physical_devices('GPU'))
        print(torch.backends.cudnn.version())
        self.data_option = _data_option

    def choise_train_dataset_option(self):
        self.LogCreator.print_and_write_log(f"Start load train data: {self.data_option}")
        if self.data_option == "custom_features":
            return self._DataPreprocessing.read_data(self._DataPreprocessing.train_custom_fetures_seleted_cleared_and_vetorized_dataset_path)
        elif self.data_option == "bert_768":
            return self._DataPreprocessing.read_data(self._DataPreprocessing.train_bert_features_selected_768_dataset_path)
        elif self.data_option == "bert_350":
            return self._DataPreprocessing.read_data(self._DataPreprocessing.train_bert_PCA_features_selected_350_dataset_path)
        else:
            print("Bad option!")
            return 0
    def choise_test_dataset_option(self):
        self.LogCreator.print_and_write_log(f"Start load test data: {self.data_option}")
        if self.data_option == "custom_features":
            return self._DataPreprocessing.read_data(self._DataPreprocessing.test_custom_fetures_seleted_cleared_and_vetorized_dataset_path)
        elif self.data_option == "bert_768":
            return self._DataPreprocessing.read_data(self._DataPreprocessing.test_bert_features_selected_768_dataset_path)
        elif self.data_option == "bert_350":
            return self._DataPreprocessing.read_data(self._DataPreprocessing.test_bert_PCA_features_selected_350_dataset_path)
        else:
            print("Bad option!")
            return 0

    def data_pathes_and_model_creation(self, option, model_name, _activation_function, _optimizer, _num_centres, _encoding_dim_AE):
        if model_name == "bert2":
            data = self._DataPreprocessing.read_data(self._DataPreprocessing.train_cleared_base_dataset_path)
            X, y = self._TrainTestDataPreproc.create_X_and_Y(data)
            tokenizer_model_name = "bert-base-uncased"
            tokenizer = BertTokenizer.from_pretrained(tokenizer_model_name)
            model, save_file_name = self._ModelNameAndPathesCreator.create_model_name_and_output_pathes(option, model_name, _activation_function, _optimizer)
            X_train, X_test, y_train, y_test = self._TrainTestDataPreproc.prepare_data_bert_2(X, y, tokenizer)
            scaler = None
        else:
            if option in [91,92,93]:
                data = self.choise_train_dataset_option()
                X, y = self._TrainTestDataPreproc.create_X_and_Y(data)
                input_size = X.shape[1]
                X_train, X_test, y_train, y_test = self._TrainTestDataPreproc.split_data_for_train_and_validation(X, y, 0.2,42)
                model, save_file_name = self._ModelNameAndPathesCreator.create_model_name_and_output_pathes(option, model_name, _activation_function, _optimizer,
                                                                                                        _num_centres ,_encoding_dim_AE,input_size)
                X_train, y_train = self._TrainTestDataPreproc.option_preprocessing(option, X_train, y_train)
                X_train = X_train.to_numpy()
                X_test = X_test.to_numpy()
                scaler = None
            elif option in [911,912,913]:
                data = self.choise_train_dataset_option()
                X, y = self._TrainTestDataPreproc.create_X_and_Y(data)
                input_size = X.shape[1]
                X_train_without_saler, X_test_without_saler, y_train, y_test = self._TrainTestDataPreproc.split_data_for_train_and_validation(
                    X, y,
                    0.2,
                    42)
                model, save_file_name = self._ModelNameAndPathesCreator.create_model_name_and_output_pathes(option,
                                                                                                            model_name,
                                                                                                            _activation_function,
                                                                                                            _optimizer,
                                                                                                            _num_centres,
                                                                                                            _encoding_dim_AE,
                                                                                                            input_size)
                scaler = self._TrainTestDataPreproc.create_scaler(X)
                #joblib.dump(scaler, f"/mnt/d/PWR/Praca magisterska/models/minmax_scaler_bert_768.pkl")
                X_train_without_saler_end, y_train = self._TrainTestDataPreproc.option_preprocessing(option,
                                                                                                     X_train_without_saler,
                                                                                                     y_train)
                X_train, X_test = self._TrainTestDataPreproc.scale_data(scaler, X_train_without_saler_end,
                                                                        X_test_without_saler)
                X_train = X_train.to_numpy()
                X_test = X_test.to_numpy()
            else:
                data = self.choise_train_dataset_option()
                X, y = self._TrainTestDataPreproc.create_X_and_Y(data)
                input_size = X.shape[1]
                X_train_without_saler, X_test_without_saler, y_train, y_test = self._TrainTestDataPreproc.split_data_for_train_and_validation(X, y,
                                                                                                                  0.2,
                                                                                                                  42)
                model, save_file_name = self._ModelNameAndPathesCreator.create_model_name_and_output_pathes(option,
                                                                                                            model_name,
                                                                                                            _activation_function,
                                                                                                            _optimizer,
                                                                                                            _num_centres,
                                                                                                            _encoding_dim_AE,
                                                                                                            input_size)
                scaler = self._TrainTestDataPreproc.create_scaler(X)
                X_train_without_saler_end, y_train = self._TrainTestDataPreproc.option_preprocessing(option, X_train_without_saler, y_train)
                X_train, X_test = self._TrainTestDataPreproc.scale_data(scaler, X_train_without_saler_end, X_test_without_saler)
                X_train = X_train.to_numpy()
                X_test = X_test.to_numpy()
        return model, save_file_name, X_train, X_test, y_train, y_test, scaler

    def print_model_summary(self, model):
        print(f"Optimizer: {model.optimizer.__class__.__name__}")
        print("Loss function:", model.loss)
        for layer in model.layers:
            if isinstance(layer, tf.keras.Sequential):
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

        # if(model.name == "autoencoder_classifier"):
        #     print(model.summary())
        # else:
        #     print(model.model.summary())
    def choise_loss(self, _loss, weights):
        loss_mapping = {
            "categorical_crossentropy": "categorical_crossentropy",
            "focal_loss": self._Optimization.focal_loss(),
            "weighted_categorical_crossentropy": self._Optimization.weighted_categorical_crossentropy(weights)
        }
        if _loss not in loss_mapping:
            print(
                f"Error: Loss function {_loss} not found! Available: {list(loss_mapping.keys())}")
            return ""
        return loss_mapping[_loss]
    def train_model(self, option, model_name, _activation_function="", _optimizer="", _loss="", _epochs = 1, _num_centres_RBFL=10, _encoding_dim_AE = 10, _model_params_string=""):
        txt = "No more config information"
        if model_name == "AE":
            txt = f"dim_AE = {_encoding_dim_AE}"
        elif model_name == "RBFL":
            txt = f"num_centres = {_num_centres_RBFL}"

        self.LogCreator.print_and_write_log(f"Train {model_name} with using {self._ModelNameAndPathesCreator.define_type_of_option(option)}\n"
                                            f"Activation_function: {_activation_function}\n"
                                            f"Optimizer: {_optimizer}\n"
                                            f"Loss: {_loss}\n"
                                            f"{txt}\n"
                                            f"{self.LogCreator.string_spit_tilds}")

        model, save_file_name, X_train, X_test, y_train, y_test, scaler = self.data_pathes_and_model_creation(option, model_name,
                                                                                                              _activation_function, _optimizer,
                                                                                                              _num_centres_RBFL, _encoding_dim_AE)
        save_file_name = save_file_name +f"_{_loss}"+f"_{_model_params_string}"
        self.LogCreator.print_and_write_log(f"Start learn model {model_name}")
        model_train_time_start = time.time()
        class_weights = self._TrainTestDataPreproc.compute_class_weights(y_train)
        loss_end = self.choise_loss(_loss, class_weights)
        if isinstance(model, models.Model) and model_name !="AE":
            if y_train.ndim == 1:
                y_train = to_categorical(y_train, num_classes=4)
                y_test = to_categorical(y_test, num_classes=4)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
            model.compile(optimizer=_optimizer, loss = loss_end, metrics = ['accuracy'])
            #self.print_model_summary(model)
            history = model.fit(X_train, y_train, epochs=_epochs, batch_size=32, validation_data=(X_test, y_test),
                      verbose=1, callbacks=[early_stopping])
            trained_epochs = len(history.epoch)
            best_epoch = early_stopping.stopped_epoch if early_stopping.stopped_epoch > 0 else trained_epochs
            self.LogCreator.print_and_write_log(
                f"Model {model_name} Training Summary:\n"
                f"Total Epochs: {_epochs}\n"
                f"Trained Epochs: {trained_epochs}\n"
                f"Best Epoch (EarlyStopping): {best_epoch if early_stopping.stopped_epoch > 0 else 'No EarlyStopping'}\n"
                f"{self.LogCreator.string_spit_stars}"
            )
            model.save(f"/mnt/d/PWR/Praca magisterska/models/{save_file_name}.keras")
            self.CM_and_ROC_creator.create_confusion_matrix(model, X_test, y_test, save_file_name)
            self.CM_and_ROC_creator.create_ROC(model, X_test, y_test, save_file_name)
            self.CM_and_ROC_creator.create_plot_traning_history(model_name, history, save_file_name)
            self.check_test_data(option, model, scaler, save_file_name)
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
            torch.save(model, f"/mnt/d/PWR/Praca magisterska/models/{save_file_name}.pth")
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
        elif model_name == "AE":
            if y_train.ndim == 1:
                y_train = to_categorical(y_train, num_classes=4)
                y_test = to_categorical(y_test, num_classes=4)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_classifier_accuracy', patience=3, restore_best_weights=True, mode = "max")
            model.compile(
                optimizer=_optimizer,
                loss={
                    "decoder": "mse",
                    "classifier": loss_end
                },
                metrics={"classifier": ["accuracy"]}
            )
            history = model.fit(
                X_train,
                {"decoder": X_train, "classifier": y_train},
                epochs=_epochs,
                batch_size=16,
                validation_data=(X_test, {"decoder": X_test, "classifier": y_test}),
                verbose=1,
                callbacks=[early_stopping]
            )
            trained_epochs = len(history.epoch)
            best_epoch = early_stopping.stopped_epoch if early_stopping.stopped_epoch > 0 else trained_epochs
            self.LogCreator.print_and_write_log(
                f"Model {model_name} Training Summary:\n"
                f"Total Epochs: {_epochs}\n"
                f"Trained Epochs: {trained_epochs}\n"
                f"Best Epoch (EarlyStopping): {best_epoch if early_stopping.stopped_epoch > 0 else 'No EarlyStopping'}\n"
                f"{self.LogCreator.string_spit_stars}"
            )
            model.save(f"/mnt/d/PWR/Praca magisterska/models/{save_file_name}.keras")
            self.CM_and_ROC_creator.create_confusion_matrix(model, X_test, y_test, save_file_name)
            self.CM_and_ROC_creator.create_ROC(model, X_test, y_test, save_file_name)
            self.CM_and_ROC_creator.create_plot_traning_history(model_name, history, save_file_name)
            self.check_test_data(option, model, scaler, save_file_name)
        else:
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()
            model.fit(X_train, y_train)
            joblib.dump(model, f"/mnt/d/PWR/Praca magisterska/models/{save_file_name}.pkl")
            self.CM_and_ROC_creator.create_confusion_matrix(model, X_test, y_test, save_file_name)
            self.CM_and_ROC_creator.create_ROC(model, X_test, y_test, save_file_name)
            self.check_test_data(option,model, scaler, save_file_name)

        model_train_time_end = time.time()
        self.LogCreator.print_and_write_log(
            f"End learn {model_name}, time to learn: {self.LogCreator.count_time(model_train_time_start, model_train_time_end):.2f} s.\n"
            f"{self.LogCreator.string_spit_stars}")


    def check_test_data(self, option, model, scaler, save_filename):
        data = self.choise_test_dataset_option()
        X, y_test = self._TrainTestDataPreproc.create_X_and_Y(data)
        if option in [91,92,93]:
            X_test_scaled = X
        else:
            X_test_scaled, _ = self._TrainTestDataPreproc.scale_data(scaler, X)
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)
        if type(model).__name__ in ['RandomForestClassifier', 'LogisticRegression',
                                    'XGBClassifier', 'MLPClassifier']:
            y_test = np.array(y_test)
        self.CM_and_ROC_creator.create_confusion_matrix(model, X_test_scaled, y_test, save_filename+"_test_data")
        self.CM_and_ROC_creator.create_ROC(model, X_test_scaled, y_test, save_filename+"_test_data")







