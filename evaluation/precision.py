"""
precision.py
------------
Precision/recall/false-positive-rate as a function of alert budget - i.e.
"if the SOC can only review the top X% of events ranked by risk score,
what do they get?" This is the curve version of the single-point
`precision_at_k` in metrics.py, used for the precision-at-alert-budget
plot and for picking a deployment threshold.
"""

import numpy as np
import pandas as pd

from .metrics import precision_at_k, false_positive_rate


def alert_budget_curve(y_true: np.ndarray, y_score: np.ndarray,
                        fractions=None) -> pd.DataFrame:
    if fractions is None:
        fractions = np.concatenate([
            np.linspace(0.001, 0.01, 10),
            np.linspace(0.02, 0.20, 19),
        ])
    n = len(y_true)
    order = np.argsort(-y_score)
    rows = []
    for f in fractions:
        k = max(1, int(np.ceil(n * f)))
        top_k_idx = order[:k]
        y_pred_at_k = np.zeros(n, dtype=int)
        y_pred_at_k[top_k_idx] = 1
        stats = precision_at_k(y_true, y_score, f)
        rows.append({
            "alert_budget_fraction": f,
            "k_events": stats["k_events"],
            "precision": stats["precision_at_k"],
            "recall": stats["recall_captured_at_k"],
            "false_positive_rate": false_positive_rate(y_true, y_pred_at_k),
        })
    return pd.DataFrame(rows)
