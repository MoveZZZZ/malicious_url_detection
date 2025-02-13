from sklearn.metrics import f1_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, mean_squared_error, mean_absolute_error
from LogSystem import LogFileCreator
class CM_and_ROC_creator:
    def __init__(self, log_filename):
        self.matrix_path = f"D:/PWR/Praca magisterska/Images/CM"
        self.ROC_path = f"D:/PWR/Praca magisterska/Images/ROC"
        self.LogCreator = LogFileCreator(log_filename)
        self.label_mapping_url = {
            'benign': 0,
            'defacement': 1,
            'phishing': 2,
            'malware': 3
        }

    def create_confusion_matrix(self, model, X_test, y_test, save_file_name):
        acc = model.score(X_test, y_test)
        pred = model.predict(X_test)
        sensitivity = recall_score(y_test, pred, average='macro', pos_label=1)
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

        num_classes = len(model.classes_)
        total_fpr, total_fnr, total_specificity = 0, 0, 0

        for i in range(len(model.classes_)):
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
                f"\nClass {model.classes_[i]} - FPR: {fpr:.4f} | FNR: {fnr:.4f} | Specificity: {specificity:.4f}")
            self.LogCreator.print_and_write_log(
                f"Class {model.classes_[i]} - FPR: {fpr:.4f} | FNR: {fnr:.4f} | Specificity: {specificity:.4f}"
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
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
        else:
            raise ValueError("Model does not have predict_proba or decision_function methods")
        y_test_binarized = label_binarize(y_test, classes=model.classes_)
        n_classes = len(model.classes_)

        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes

        macro_roc_auc = auc(all_fpr, mean_tpr)

        # Plot all ROC curves
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label=f"Class {model.classes_[i]} (AUC = {roc_auc[i]:.2f})")

        plt.plot(all_fpr, mean_tpr, color='navy', linestyle='--', lw=2,
                 label=f"Macro-average (AUC = {macro_roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for Multiclass Classification")
        plt.legend(loc="lower right")

        plt.savefig(f"{self.ROC_path}/{save_file_name}_roc_curve.png", bbox_inches='tight')
        plt.close()