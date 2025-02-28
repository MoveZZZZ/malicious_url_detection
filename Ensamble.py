from TrainTestDataPreprocessing import TrainTestDataPreprocessing
from DataPreprocessing import DataPreprocessing
from ConfusionMatrixAndRocCreator import CM_and_ROC_creator
from LogSystem import LogFileCreator
from TrainModels import TrainModels
import os
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, \
    mean_absolute_error
import pandas as pd
from CustomModels import DeepMLP_3,DeepMLP_5,DeepMLP_7, AutoencoderClassifier
class ModelEnsembler:
    def __init__(self):
        self.log_name = "ensamble"
        self.TrainTestPreproc = TrainTestDataPreprocessing(self.log_name)
        self.DataPreproc = DataPreprocessing(self.log_name)
        self.ROC_Matrix = CM_and_ROC_creator(self.log_name)
        self.DataLog = LogFileCreator(self.log_name)
        self.min_max_scaler = None
        self.models_files = None
        self.models = []
        self.folder_path = "D:/PWR/Praca magisterska/models/1_test"

    def load_models_from_folder(self):
        custom_objects = {
            "AutoencoderClassifier": AutoencoderClassifier,
            "DeepMLP_3": DeepMLP_3,
            "DeepMLP_5": DeepMLP_5,
            "DeepMLP_7": DeepMLP_7
        }
        self.models_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith(".keras")])
        self.models = [load_model(os.path.join(self.folder_path, file), custom_objects=custom_objects) for file in self.models_files]
        self.DataLog.print_and_write_log(f"Loaded {len(self.models)}  models: {self.models_files}")
        tm = TrainModels(self.log_name, "123")
        tm.print_model_summary(self.models[0])
    def load_scaler(self):
        scaler_path = "D:/PWR/Praca magisterska/models/1_test/minmax_scaler_bert_768_browser_np_features_full_data_scaled.pkl"
        scaler = joblib.load(scaler_path)
        return scaler

    def stacking(self, X_test, y_test):
        predictions = np.array([model.predict(X_test) for model in self.models])
        n_samples = X_test.shape[0]
        predictions_meta = predictions.transpose(1, 0, 2).reshape(n_samples, -1)
        X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(
            predictions_meta, y_test, test_size=0.2, random_state=42
        )
        y_train_meta = np.argmax(y_train_meta, axis=1)
        y_val_meta = np.argmax(y_val_meta, axis=1)
        meta_model = LogisticRegression(max_iter=1000, solver='liblinear')
        meta_model.fit(X_train_meta, y_train_meta)
        final_predictions = meta_model.predict(X_val_meta)
        accuracy = accuracy_score(y_val_meta, final_predictions)
        print(f"Stacking model accuracy: {accuracy:.4f}")
        final_predictions_all = meta_model.predict(predictions_meta)
        return final_predictions_all

    def test(self):
        self.load_models_from_folder()
        self.min_max_scaler = self.load_scaler()
        test_data = self.DataPreproc.read_data(self.DataPreproc.test_bert_features_selected_768_selenium_and_more_rare_class_dataset_path)
        X_test, y_test = self.TrainTestPreproc.create_X_and_Y(test_data)
        X_test_scaled, _ = self.TrainTestPreproc.scale_data(self.min_max_scaler, X_test)
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)
        y_test = to_categorical(y_test, num_classes=4)

        predicted = self.stacking(X_test_scaled, y_test)
        file_name = "ensamble_test"
        self.create_matrix(y_test, predicted, file_name)

    def create_matrix(self, y, y_pred, filename):
        matrix_path = "D:/PWR/Praca magisterska/models/1_test/images"
        if y.ndim > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)

        y_series = pd.Series(y)
        print(y_series.value_counts())
        acc = np.mean(y == y_pred)
        if len(np.unique(y)) == 2:
            sensitivity = recall_score(y, y_pred, average='macro', pos_label=1)
            f1 = f1_score(y, y_pred, average='macro')
        else:
            sensitivity = recall_score(y, y_pred, average='macro')
            f1 = f1_score(y, y_pred, average='macro')
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(15, 15))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        print("Matrix (without round):")
        print(cm)

        disp.plot(ax=ax, cmap='Blues', colorbar=True)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                disp.text_[i, j].set_text(f"{value:,}")
        epe = mean_squared_error(y, y_pred)
        model_error = mean_absolute_error(y, y_pred)
        metrics_text = (f"Acc: {acc:.4f} | Sens: {sensitivity:.4f} | F1: {f1:.4f}\n"
                        f"EPE (MSE): {epe:.4f} | Błąd (MAE): {model_error:.4f}\n")
        self.DataLog.print_and_write_log(
            f"Acc: {acc:.4f} | Sens: {sensitivity:.4f} | F1: {f1:.4f}\n"
            f"EPE (MSE): {epe:.4f} | Błąd (MAE): {model_error:.4f}\n"
            f"{self.DataLog.string_spit_stars}")
        if len(np.unique(y)) == 2:
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics_text += f"FPR: {fpr:.4f} | FNR: {fnr:.4f} | Specificity: {specificity:.4f}"
            self.DataLog.print_and_write_log(
                f"FPR: {fpr:.4f} | FNR: {fnr:.4f} | Specificity: {specificity:.4f}\n"
                f"{self.DataLog.string_spit_stars}")
        else:
            for i in range(len(np.unique(y))):
                tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
                fp = cm[:, i].sum() - cm[i, i]
                fn = cm[i, :].sum() - cm[i, i]
                tp = cm[i, i]

                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                metrics_text += (
                    f"\nClass {np.unique(y)[i]} - FPR: {fpr:.4f} | FNR: {fnr:.4f} | Specificity: {specificity:.4f}")
                self.DataLog.print_and_write_log(
                    f"\nClass {np.unique(y)[i]} - FPR: {fpr:.4f} | FNR: {fnr:.4f} | Specificity: {specificity:.4f}\n"
                    f"{self.DataLog.string_spit_stars}")

        plt.subplots_adjust(bottom=0.2)
        plt.figtext(0.5, 0.15, metrics_text, fontsize=10, ha='center', va='top', color="red")
        plt.savefig(f"{matrix_path}/{filename}_confusion_matrix.png", bbox_inches='tight')
        plt.close()


ens = ModelEnsembler()
ens.test()