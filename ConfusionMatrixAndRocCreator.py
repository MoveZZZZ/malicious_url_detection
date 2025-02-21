from sklearn.metrics import f1_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, mean_squared_error, mean_absolute_error,accuracy_score
from LogSystem import LogFileCreator
import tensorflow as tf
from CustomModels import *

class CM_and_ROC_creator:
    def __init__(self, log_filename):
        # self.matrix_path = f"D:/PWR/Praca magisterska/Images/CM"
        # self.ROC_path = f"D:/PWR/Praca magisterska/Images/ROC"
        # self.train_history_path = f"D:/PWR/Praca magisterska/Images/TH"

        self.train_history_path = f"/mnt/d/PWR/Praca magisterska/Images/TH"
        self.matrix_path = f"/mnt/d/PWR/Praca magisterska/Images/CM"
        self.ROC_path = f"/mnt/d/PWR/Praca magisterska/Images/ROC"

        self.LogCreator = LogFileCreator(log_filename)
        self.label_mapping_url = {
            'benign': 0,
            'defacement': 1,
            'phishing': 2,
            'malware': 3
        }

    def create_confusion_matrix(self, model, X_test, y_test, save_file_name):
        if type(model).__name__ == "TFBertForSequenceClassification":
            pred_probs = model.predict([X_test["input_ids"], X_test["attention_mask"]])
            logits = pred_probs.logits
            print("Shape of logits:", logits.shape)
            if len(logits.shape) == 2:
                pred = np.argmax(logits, axis=1)
            else:
                print("Predictions seem to be 1-dimensional, please check the model output.")
                pred = logits
            try:
                acc = model.score(X_test, y_test)
            except:
                acc = accuracy_score(y_test, pred)
        elif hasattr(model, "evaluate"):
            if isinstance(model, AutoencoderClassifier):
                y_pred_dict = model.predict(X_test)
                y_pred = y_pred_dict["classifier"]
                pred = np.argmax(y_pred, axis=1)
                eval_results = model.evaluate(X_test, {"decoder": X_test, "classifier": y_test}, verbose=0)
                if "classifier_accuracy" in model.metrics_names:
                    acc = eval_results[model.metrics_names.index("classifier_accuracy")]
                else:
                    y_pred_dict = model.predict(X_test)
                    y_pred = np.argmax(y_pred_dict["classifier"], axis=1)
                    y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
                    acc = accuracy_score(y_true, y_pred)
            else:
                acc = model.evaluate(X_test, y_test, verbose=0)[1]
                pred = np.argmax(model.predict(X_test), axis=1)
                if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                    y_test = np.argmax(y_test, axis=1)
        else:
            pred = model.predict(X_test)
            try:
                acc = model.score(X_test, y_test)
            except:
                acc = accuracy_score(y_test, pred)

        num_classes = 4
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)

        sensitivity = recall_score(y_test, pred, average='macro')
        f1 = f1_score(y_test, pred, average='macro')
        cm = confusion_matrix(y_test, pred, labels=list(self.label_mapping_url.values()))
        fig, ax = plt.subplots(figsize=(15, 15))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(self.label_mapping_url.keys()))

        disp.plot(ax=ax, cmap='Blues', colorbar=True)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                disp.text_[i, j].set_text(f"{value:,}")

        epe = mean_squared_error(y_test, pred)
        model_error = mean_absolute_error(y_test, pred)
        metrics_text = f"Acc: {acc:.4f} | Sens: {sensitivity:.4f} | F1: {f1:.4f}\nEPE (MSE): {epe:.4f} | Błąd (MAE): {model_error:.4f}\n"

        total_fpr, total_fnr, total_specificity = 0, 0, 0
        for i in range(num_classes):
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fp = cm[:, i].sum() - cm[i, i]
            fn = cm[i, :].sum() - cm[i, i]
            tp = cm[i, i]

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            total_fpr += fpr
            total_fnr += fnr
            total_specificity += specificity
            metrics_text += (
                f"\nClass {i} - FPR: {fpr:.4f} | FNR: {fnr:.4f} | Specificity: {specificity:.4f}")
            self.LogCreator.print_and_write_log(
                f"Class {i} - FPR: {fpr:.4f} | FNR: {fnr:.4f} | Specificity: {specificity:.4f}"
                f"\n{self.LogCreator.string_spit_stars}")

        average_fpr = total_fpr / num_classes
        average_fnr = total_fnr / num_classes
        average_specificity = total_specificity / num_classes
        self.LogCreator.print_and_write_log(
            f"Average metrics\n"
            f"Acc: {acc:.4f} | Sens: {sensitivity:.4f} | F1: {f1:.4f}\nEPE (MSE): {epe:.4f} | Błąd (MAE): {model_error:.4f}\n"
            f"FPR: {average_fpr:.4f} | FNR: {average_fnr:.4f} | Specificity: {average_specificity:.4f}"
            f"\n{self.LogCreator.string_spit_stars}")
        plt.subplots_adjust(bottom=0.2)
        plt.title(f"Confusion matrix for: {type(model).__name__}")
        plt.figtext(0.5, 0.15, metrics_text, fontsize=10, ha='center', va='top', color="red")
        plt.savefig(f"{self.matrix_path}/{save_file_name}_confusion_matrix.png", bbox_inches='tight')
        plt.close()

    def create_ROC(self, model, X_test, y_test, save_file_name):
        if isinstance(model, AutoencoderClassifier):
            y_prob = model.predict(X_test)["classifier"]
        elif hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
        elif hasattr(model, "predict"):
            y_prob = model.predict(X_test)
        else:
            raise ValueError("Unsupported model type")

        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test_binarized = y_test
        else:
            classes = np.unique(y_test)
            y_test_binarized = label_binarize(y_test, classes=classes)
        n_classes = y_test_binarized.shape[1]
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        macro_roc_auc = auc(all_fpr, mean_tpr)
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
        plt.plot(all_fpr, mean_tpr, color='navy', linestyle='--', lw=2,
                 label=f"Macro-average (AUC = {macro_roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for Multiclass Classification for {type(model).__name__}")
        plt.legend(loc="lower right")
        plt.savefig(f"{self.ROC_path}/{save_file_name}_roc_curve.png", bbox_inches='tight')
        plt.close()

    def create_plot_traning_history(self, model_name, history, save_file_name):
        if model_name == "tabnet":
            print("Tabnet unsupported history!")
            return
        if model_name == "AE":
            train_loss = history.history.get('loss', [])
            val_loss = history.history.get('val_loss', [])
            train_decoder_loss = history.history.get('decoder_loss', [])
            val_decoder_loss = history.history.get('val_decoder_loss', [])
            train_classifier_acc = history.history.get('classifier_accuracy', [])
            val_classifier_acc = history.history.get('val_classifier_accuracy', [])
            train_acc, val_acc = [], []
        else:
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']
            train_acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            train_decoder_loss, val_decoder_loss = [], []
            train_classifier_acc, val_classifier_acc = [], []


        if len(train_loss) == 0 or len(val_loss) == 0:
            print("No loss data found, skipping plot generation.")
            return

        epochs = range(1, len(train_loss) + 1)
        if len(train_loss) > 0 and len(val_loss) > 0:
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 2, 1)
            plt.plot(epochs, train_loss, 'r', label='Training Loss')
            plt.plot(epochs, val_loss, 'b', label='Validation Loss')
            plt.title(f'Training and Validation Loss for {model_name}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

        if len(train_acc) > 0 and len(val_acc) > 0:
            plt.subplot(2, 2, 2)
            plt.plot(epochs, train_acc, 'r', label='Training accuracy')
            plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
            plt.title(f'Training and Validation Accuracy for {model_name}')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

        if len(train_decoder_loss) > 0 and len(val_decoder_loss) > 0:
            plt.subplot(2, 2, 3)
            plt.plot(epochs, train_decoder_loss, 'r', label='Training Decoder Loss')
            plt.plot(epochs, val_decoder_loss, 'b', label='Validation Decoder Loss')
            plt.title(f'Decoder Loss for {model_name}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
        if len(train_classifier_acc) > 0 and len(val_classifier_acc) > 0:
            plt.subplot(2, 2, 4)
            plt.plot(epochs, train_classifier_acc, 'r', label='Training Classifier Accuracy')
            plt.plot(epochs, val_classifier_acc, 'b', label='Validation Classifier Accuracy')
            plt.title(f'Classifier Accuracy for {model_name}')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.train_history_path}/{save_file_name}_th.png", bbox_inches='tight')
        plt.close()

