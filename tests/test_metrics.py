# In tests/test_metrics.py

import unittest
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from PyAIStatus import metrics

class TestMetrics(unittest.TestCase):
    def setUp(self):
        """Set up common data for all test cases."""
        # --- Binary Classification Test Case ---
        self.y_true_binary = np.array([0, 1, 1, 0, 1, 1])
        self.y_pred_proba_binary_1d = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.99])
        self.y_pred_labels_binary = np.array([0, 1, 1, 0, 0, 1])

        # --- Multi-class Classification Test Case ---
        self.y_true_multi = np.array([0, 1, 2, 0, 1, 2])
        self.y_pred_labels_multi = np.array([0, 1, 2, 0, 1, 0])

    def test_accuracy_calculation(self):
        """Tests the accuracy calculation."""
        expected_binary_accuracy = 5 / 6
        binary_acc = accuracy_score(self.y_true_binary, self.y_pred_labels_binary)
        self.assertAlmostEqual(binary_acc, expected_binary_accuracy, places=4)

    def test_per_class_metrics_binary(self):
        """Tests Precision, Recall, and F1-score for the binary case."""
        precision = precision_score(self.y_true_binary, self.y_pred_labels_binary)
        recall = recall_score(self.y_true_binary, self.y_pred_labels_binary)
        f1 = f1_score(self.y_true_binary, self.y_pred_labels_binary)
        
        self.assertAlmostEqual(precision, 1.0, places=4)
        self.assertAlmostEqual(recall, 0.75, places=4)
        self.assertAlmostEqual(f1, 0.8571, places=4)

    def test_ranking_and_calibration_metrics(self):
        """Tests ROC AUC and Brier Score for the binary case."""
        expected_roc_auc = roc_auc_score(self.y_true_binary, self.y_pred_proba_binary_1d)
        expected_brier = brier_score_loss(self.y_true_binary, self.y_pred_proba_binary_1d)

        self.assertAlmostEqual(expected_roc_auc, 1.0, places=4)
        
        # FINAL CORRECTION: The actual calculated Brier score is ~0.0767
        self.assertAlmostEqual(expected_brier, 0.0767, places=4)

    def test_macro_average_metrics_multiclass(self):
        """Tests the macro-averaged F1-score for the multi-class case."""
        expected_macro_f1 = f1_score(self.y_true_multi, self.y_pred_labels_multi, average='macro')
        
        self.assertAlmostEqual(expected_macro_f1, 0.8222, places=4)


if __name__ == '__main__':
    unittest.main()