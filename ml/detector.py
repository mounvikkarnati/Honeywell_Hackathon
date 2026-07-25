"""
detector.py
-----------
The Detection Model (Deliverable #3). Isolation Forest is the required
Tier-1 baseline: it's the right tool here because it doesn't need any
labeled anomalies to train (it isolates points that are "few and
different" in feature space) - matching the real-world constraint that
true intrusions are a tiny, mostly-unknown-shape fraction of events.

Training data: the FULL feature matrix, entirely unsupervised, contamination
set close to (but not exactly) the true injected rate so the model isn't
implicitly told the exact answer.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import FEATURE_COLUMNS, get_X


class AnomalyDetector:
    def __init__(self, contamination: float = 0.02, n_estimators: int = 300,
                 random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_columns = FEATURE_COLUMNS

    def fit(self, feats: pd.DataFrame):
        X = get_X(feats)
        self.model.fit(X)
        return self

    def score(self, feats: pd.DataFrame) -> pd.Series:
        """Higher score = more anomalous (0-100 scale for the dashboard),
        derived from sklearn's raw decision_function (higher = more normal
        in sklearn's convention, hence the sign flip)."""
        X = get_X(feats)
        raw = self.model.decision_function(X)  # higher = more normal
        anomaly_raw = -raw                     # higher = more anomalous

        # Min-max scale to 0-100 for a human-readable risk score. Uses the
        # fit-time raw score range would be more principled for a live
        # system; here we scale per-call for simplicity, which is fine
        # for a single evaluation batch but should be pinned to a fixed
        # reference range if this scores a live stream incrementally.
        lo, hi = anomaly_raw.min(), anomaly_raw.max()
        scaled = 100 * (anomaly_raw - lo) / max(hi - lo, 1e-9)
        return pd.Series(scaled, index=feats.index, name="risk_score")

    def predict_outlier(self, feats: pd.DataFrame) -> pd.Series:
        """-1 = outlier (anomaly), 1 = inlier (normal), per sklearn convention."""
        X = get_X(feats)
        return pd.Series(self.model.predict(X), index=feats.index, name="is_outlier")
