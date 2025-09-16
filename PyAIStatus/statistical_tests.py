# PyAIStatus/PyAIStatus/statistical_tests.py

import numpy as np
from sklearn.metrics import accuracy_score

# Define a fixed random seed for reproducibility of the bootstrap test.
RANDOM_SEED = 42

def _get_labels_from_proba(y_pred_proba: np.ndarray) -> np.ndarray:
    """Helper function to convert probabilities to labels, handling both binary and categorical cases."""
    if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 1:
        # Binary case (sigmoid output)
        return (y_pred_proba > 0.5).astype(int).squeeze()
    else:
        # Categorical case (softmax output)
        return np.argmax(y_pred_proba, axis=1)

def compare_models(y_true: np.ndarray, 
                   y_pred_proba_model: np.ndarray, 
                   y_pred_proba_baseline: np.ndarray, 
                   metric_func=accuracy_score, 
                   n_bootstraps: int = 1000) -> dict:
    """
    Performs a paired bootstrap test to compare two models (Task 11).

    This test determines if the difference in performance between the main model and the
    baseline is statistically significant.

    Args:
        y_true: Ground truth labels (can be a list or numpy array).
        y_pred_proba_model: Predicted probabilities from the main model.
        y_pred_proba_baseline: Predicted probabilities from the baseline model.
        metric_func: The metric function to compare (e.g., accuracy_score).
        n_bootstraps: The number of bootstrap iterations to perform.

    Returns:
        A dictionary containing the p-value, observed difference, and confidence interval.
    """
    print("\n--- Performing Statistical Test (Paired Bootstrap) ---")
    
    # --- THIS IS THE FIX ---
    # Convert y_true to a NumPy array to allow for advanced indexing with 'indices'.
    y_true = np.array(y_true)
    # --- END FIX ---

    # Convert probabilities to labels for both models
    y_pred_labels_model = _get_labels_from_proba(y_pred_proba_model)
    y_pred_labels_baseline = _get_labels_from_proba(y_pred_proba_baseline)

    # 1. Calculate the observed difference in performance on the actual test set
    metric_model = metric_func(y_true, y_pred_labels_model)
    metric_baseline = metric_func(y_true, y_pred_labels_baseline)
    observed_difference = metric_model - metric_baseline

    # 2. Perform bootstrapping
    rng = np.random.RandomState(RANDOM_SEED)
    bootstrap_differences = []
    n_samples = len(y_true)

    for i in range(n_bootstraps):
        # Create a bootstrap sample by sampling indices with replacement
        indices = rng.randint(0, n_samples, n_samples)
        
        # Ensure the sample is valid (has more than one class)
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        # Calculate metric on the bootstrap sample for both models
        # This now works because y_true is a NumPy array
        metric_model_sample = metric_func(y_true[indices], y_pred_labels_model[indices])
        metric_baseline_sample = metric_func(y_true[indices], y_pred_labels_baseline[indices])
        
        # Store the difference
        bootstrap_differences.append(metric_model_sample - metric_baseline_sample)
    
    bootstrap_differences = np.array(bootstrap_differences)

    # 3. Calculate the p-value
    # We test how often the bootstrapped difference is <= 0 (i.e., baseline is better or equal)
    # A two-tailed test
    p_value_one_side = np.mean(bootstrap_differences <= 0)
    p_value = 2 * min(p_value_one_side, 1 - p_value_one_side)

    # 4. Calculate the 95% confidence interval for the difference
    lower_bound = np.percentile(bootstrap_differences, 2.5)
    upper_bound = np.percentile(bootstrap_differences, 97.5)
    
    results = {
        "p_value": p_value,
        "observed_difference": observed_difference,
        "confidence_interval": (lower_bound, upper_bound),
    }
    
    print(f"  Statistical test complete.")
    print(f"  - Observed Accuracy Difference (Model - Baseline): {observed_difference:.4f}")
    print(f"  - 95% CI of Difference: ({lower_bound:.4f}, {upper_bound:.4f})")
    print(f"  - P-value: {p_value:.4f}")
    
    return results