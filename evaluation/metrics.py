"""
metrics.py
----------
Core metric computations for the anomaly detector and attack-type
classifier. Every number that ends up in the report or dashboard should
be computed by calling into this module - never hand-calculated inline,
so the same computation is used everywhere and can't silently drift.

Inputs are plain numpy/pandas objects (no sklearn-object coupling) so this
module works for any model, not just the Isolation Forest.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve,
)


def binary_labels_from_ground_truth(labels: pd.Series) -> np.ndarray:
    """Ground truth 'label' column -> binary y_true (1 = any anomaly, 0 = normal)."""
    return (labels != "normal").astype(int).to_numpy()


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_fraction: float = 0.01) -> dict:
    """
    Precision at a realistic analyst alert budget (e.g. top 1% of events
    by risk score) - this is one of the six named evaluation criteria, so
    it gets its own first-class function rather than being folded into a
    generic "compute everything" call.
    """
    n = len(y_true)
    k = max(1, int(np.ceil(n * k_fraction)))
    order = np.argsort(-y_score)
    top_k_idx = order[:k]
    tp = y_true[top_k_idx].sum()
    precision = tp / k
    recall_of_all_anomalies = tp / max(y_true.sum(), 1)
    return {
        "k_fraction": k_fraction,
        "k_events": int(k),
        "true_positives_in_top_k": int(tp),
        "precision_at_k": float(precision),
        "recall_captured_at_k": float(recall_of_all_anomalies),
    }


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(fp / max(fp + tn, 1))


def binary_detection_metrics(y_true: np.ndarray, y_score: np.ndarray,
                              y_pred: np.ndarray, alert_budget_fractions=(0.01, 0.02, 0.05)) -> dict:
    """
    Full metric bundle for the binary anomaly-detection task.
      y_true  : 0/1 ground truth (1 = anomaly)
      y_score : continuous risk score, higher = more anomalous
      y_pred  : 0/1 hard prediction at whatever threshold the model used
    """
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "false_positive_rate": false_positive_rate(y_true, y_pred),
        "n_total": int(len(y_true)),
        "n_anomalies_true": int(y_true.sum()),
        "n_flagged": int(y_pred.sum()),
    }
    metrics["precision_at_k"] = {
        f"top_{int(f*100)}pct": precision_at_k(y_true, y_score, f)
        for f in alert_budget_fractions
    }
    return metrics


def classification_report_multiclass(y_true_type: pd.Series, y_pred_type: pd.Series) -> dict:
    """
    Attack-TYPE classification metrics, restricted to rows that are truly
    anomalous (comparing 'which attack category' predictions only makes
    sense there - a normal row has no true attack type to get right).
    """
    from sklearn.metrics import classification_report
    report = classification_report(y_true_type, y_pred_type, output_dict=True, zero_division=0)
    return report
