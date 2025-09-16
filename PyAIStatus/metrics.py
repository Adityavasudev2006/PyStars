# PyAIStatus/PyAIStatus/metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

def compute_all_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, class_names: list) -> dict:
   
    y_true = np.array(y_true)

    if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 1:
        print("Model output is binary. Converting to categorical format for metric calculation.")
        y_pred_proba = np.hstack([1 - y_pred_proba, y_pred_proba])

    y_pred_labels = np.argmax(y_pred_proba, axis=1)
    num_classes = len(class_names)

    accuracy = accuracy_score(y_true, y_pred_labels)
    
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_labels, average='macro', zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_labels, average='micro', zero_division=0
    )

    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred_labels, average=None, zero_division=0, labels=np.arange(len(class_names))
    )

    y_true_one_hot = label_binarize(y_true, classes=np.arange(num_classes))
    per_class_roc_auc = {}
    
    if num_classes == 2:
        positive_class_probs = y_pred_proba[:, 1]
        macro_roc_auc = roc_auc_score(y_true, positive_class_probs)
        per_class_roc_auc[class_names[1]] = macro_roc_auc
    else:
        macro_roc_auc = roc_auc_score(y_true_one_hot, y_pred_proba, multi_class='ovr', average='macro')
        for i, name in enumerate(class_names):
             per_class_roc_auc[name] = roc_auc_score(y_true_one_hot[:, i], y_pred_proba[:, i])

    per_class_pr_auc = {}
    if num_classes == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
        per_class_pr_auc[class_names[1]] = auc(recall, precision)
        precision, recall, _ = precision_recall_curve(1 - y_true, y_pred_proba[:, 0])
        per_class_pr_auc[class_names[0]] = auc(recall, precision)
    else:
        for i, name in enumerate(class_names):
            precision, recall, _ = precision_recall_curve(y_true_one_hot[:, i], y_pred_proba[:, i])
            per_class_pr_auc[name] = auc(recall, precision)

    metrics_results = {
        "overall": {"accuracy": accuracy},
        "macro_average": {"precision": macro_precision, "recall": macro_recall, "f1_score": macro_f1},
        "micro_average": {"precision": micro_precision, "recall": micro_recall, "f1_score": micro_f1},
        "per_class": {
            name: {
                "precision": per_class_precision[i], "recall": per_class_recall[i],
                "f1_score": per_class_f1[i], "support": int(per_class_support[i]),
                "roc_auc": per_class_roc_auc.get(name, float('nan')),
                "pr_auc": per_class_pr_auc.get(name, float('nan')),
            } for i, name in enumerate(class_names)
        },
        "ranking": {"macro_roc_auc": macro_roc_auc}
    }
    print("\nMetrics Calculation Successful.")
    return metrics_results

def get_bootstrapped_ci(y_true: np.ndarray, y_pred_proba: np.ndarray, metric_func=accuracy_score, n_bootstraps: int = 1000) -> tuple:
    """
    Computes 95% confidence intervals for a given metric using bootstrapping.

    Args:
        y_true: The true labels.
        y_pred_proba: The predicted probabilities from the model.
        metric_func: The metric function to evaluate (e.g., accuracy_score).
        n_bootstraps: The number of bootstrap samples to create.

    Returns:
        A tuple containing the lower and upper bounds of the 95% CI.
    """
    # Ensure y_true is a numpy array for robust indexing
    y_true = np.array(y_true)
    
    rng = np.random.RandomState(42)
    bootstrapped_scores = []
    n_samples = len(y_true)

    for i in range(n_bootstraps):
        # Create a bootstrap sample of indices
        indices = rng.randint(0, n_samples, n_samples)
        
        # Ensure the sample is not degenerate (contains only one class)
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        # Get the bootstrap sample of predictions
        boot_pred_proba = y_pred_proba[indices]
        
        # Convert probabilities to labels for this specific sample
        boot_pred_labels = np.argmax(boot_pred_proba, axis=1)
        
        # Calculate the metric for this bootstrap sample
        score = metric_func(y_true[indices], boot_pred_labels)
        bootstrapped_scores.append(score)
        
    # Calculate the 95% confidence interval
    lower_bound = np.percentile(bootstrapped_scores, 2.5)
    upper_bound = np.percentile(bootstrapped_scores, 97.5)
    
    return (lower_bound, upper_bound)

def compute_calibration_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> dict:
    y_true = np.array(y_true) # Also add this safety check here
    if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 1:
        brier_score = brier_score_loss(y_true, y_pred_proba.squeeze())
        y_pred_proba = np.hstack([1 - y_pred_proba, y_pred_proba])
    else:
        y_true_one_hot = label_binarize(y_true, classes=range(y_pred_proba.shape[1]))
        brier_score = np.mean(np.sum((y_pred_proba - y_true_one_hot)**2, axis=1))

    confidences = np.max(y_pred_proba, axis=1)
    
    mean_confidence = np.mean(confidences)

    y_pred_labels = np.argmax(y_pred_proba, axis=1)
    accuracies = (y_pred_labels == y_true)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    mean_confidence = np.mean(confidences)

    return {
        "brier_score": brier_score,
        "expected_calibration_error": ece,
        "mean_confidence": mean_confidence
    }