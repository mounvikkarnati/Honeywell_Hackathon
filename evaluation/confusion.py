"""
confusion.py
------------
Confusion matrices for both the binary detection task and the multi-class
attack-type classification task. Kept separate from metrics.py so the
matrix DATA (this file) is decoupled from the matrix VISUALIZATION
(plots.py) - the report generator needs the former, the dashboard/slides
need the latter, and they shouldn't have to import each other.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "matrix": cm.tolist(),
        "labels": ["normal", "anomaly"],
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def multiclass_confusion(y_true_type: pd.Series, y_pred_type: pd.Series) -> dict:
    labels = sorted(set(y_true_type.unique()) | set(y_pred_type.unique()))
    cm = confusion_matrix(y_true_type, y_pred_type, labels=labels)
    return {
        "matrix": cm.tolist(),
        "labels": labels,
    }
