# PyAIStatus/PyAIStatus/plots.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

# --- Helper function to save plot to a base64 string ---
def _save_plot_to_base64() -> str:
    """Saves the current matplotlib plot to a base64 encoded string."""
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    # The returned string is already in the format required by HTML <img> tags
    img_str = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    plt.close() # Close the plot to free up memory
    return img_str

# --- Helper function to handle binary vs categorical outputs ---
def _prepare_predictions(y_pred_proba):
    """Ensures prediction array is in categorical format."""
    if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 1:
        return np.hstack([1 - y_pred_proba, y_pred_proba])
    return y_pred_proba

# --- Individual Plotting Functions (Now returning base64 strings) ---

def _plot_confusion_matrix(y_true, y_pred_proba, class_names):
    """Generates a confusion matrix plot and returns it as a base64 string."""
    y_pred_labels = np.argmax(_prepare_predictions(y_pred_proba), axis=1)
    cm = confusion_matrix(y_true, y_pred_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return _save_plot_to_base64()

def _plot_roc_curves(y_true, y_pred_proba, class_names):
    """Generates ROC curves and returns the plot as a base64 string."""
    y_pred_proba = _prepare_predictions(y_pred_proba)
    num_classes = len(class_names)
    
    plt.figure(figsize=(8, 6))
    
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve for {class_names[1]} (area = {roc_auc:0.2f})')
    else:
        y_true_one_hot = label_binarize(y_true, classes=np.arange(num_classes))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve for {class_names[i]} (area = {roc_auc:0.2f})')
            
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    return _save_plot_to_base64()

def _plot_pr_curves(y_true, y_pred_proba, class_names):
    """Generates Precision-Recall curves and returns the plot as a base64 string."""
    y_pred_proba = _prepare_predictions(y_pred_proba)
    num_classes = len(class_names)

    plt.figure(figsize=(8, 6))

    if num_classes == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'PR curve for {class_names[1]} (area = {pr_auc:0.2f})')
    else:
        y_true_one_hot = label_binarize(y_true, classes=np.arange(num_classes))
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_true_one_hot[:, i], y_pred_proba[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f'PR curve for {class_names[i]} (area = {pr_auc:0.2f})')
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    return _save_plot_to_base64()

def _plot_calibration_curve(y_true, y_pred_proba):
    """Generates a calibration curve plot and returns it as a base64 string."""
    plt.figure(figsize=(8, 6))
    
    y_pred_proba = _prepare_predictions(y_pred_proba)
    if y_pred_proba.shape[1] == 2:
        y_prob = y_pred_proba[:, 1]
    else:
        y_prob = np.max(y_pred_proba, axis=1)

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')

    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.ylabel('Fraction of Positives (True Frequency)')
    plt.xlabel('Mean Predicted Value (Confidence)')
    plt.legend()
    plt.grid(True)
    return _save_plot_to_base64()

# --- Main function to generate all plots ---

def generate_all_plots(y_true, y_pred_proba, class_names: list, output_dir: str) -> dict:
    """
    Main function to generate all required plots as base64 strings.
    """
    print("\n--- Generating Plots ---")
    y_true = np.array(y_true)
    os.makedirs(output_dir, exist_ok=True) # Still useful for other outputs like attributions

    plot_data = {
        "confusion_matrix": _plot_confusion_matrix(y_true, y_pred_proba, class_names),
        "roc_curves": _plot_roc_curves(y_true, y_pred_proba, class_names),
        "pr_curves": _plot_pr_curves(y_true, y_pred_proba, class_names),
        "calibration_curve": _plot_calibration_curve(y_true, y_pred_proba),
    }
    
    print("All plots generated successfully.")
    return plot_data