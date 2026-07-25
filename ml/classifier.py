"""
classifier.py
-------------
The Anomaly Classification model (Deliverable #4): given an event already
flagged as anomalous, which attack category does it resemble?

RandomForest chosen over a heavier sequence model here deliberately - per
the hackathon plan's Tier 1/2 prioritization, a single strong classical
model that's fully working beats a fancier model that's half-debugged at
demo time. Trained ONLY on rows with a non-"normal" label (classifying
attack type is meaningless for normal traffic), using the same feature
set the detector uses so both models are consistent about what
"suspicious-looking" means.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .features import FEATURE_COLUMNS, get_X


class AttackTypeClassifier:
    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",   # attack types are themselves imbalanced
                                       # (brute_force >> device_spoofing in our
                                       # generator's injection mix)
        )
        self.feature_columns = FEATURE_COLUMNS
        self.classes_ = None

    def fit(self, feats: pd.DataFrame, labels: pd.Series):
        """feats/labels must already be filtered to non-'normal' rows only."""
        X = get_X(feats)
        self.model.fit(X, labels)
        self.classes_ = self.model.classes_
        return self

    def predict(self, feats: pd.DataFrame) -> pd.Series:
        X = get_X(feats)
        preds = self.model.predict(X)
        return pd.Series(preds, index=feats.index, name="predicted_type")

    def predict_proba_top(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Returns the top predicted class and its probability - useful
        for the dashboard/explainability layer ('87% confidence this
        resembles lateral_movement')."""
        X = get_X(feats)
        proba = self.model.predict_proba(X)
        top_idx = proba.argmax(axis=1)
        top_class = self.model.classes_[top_idx]
        top_proba = proba[range(len(proba)), top_idx]
        return pd.DataFrame({
            "predicted_type": top_class,
            "predicted_type_confidence": top_proba,
        }, index=feats.index)

    def feature_importances(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.feature_columns).sort_values(ascending=False)
