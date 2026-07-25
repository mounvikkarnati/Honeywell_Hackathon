"""
explain.py
----------
Basic explainability layer (Phase 4 version - a fuller SHAP-based version
is planned for Phase 5). For each flagged alert, ranks which engineered
features deviated most from the global "normal" distribution and turns
that into a human-readable reason string, e.g.:

    "Flagged due to new device fingerprint + unusually high implied
     travel speed (impossible-travel pattern)."

Standardization uses feature statistics computed from NORMAL rows only
(not the whole dataset) - so "deviation" genuinely means "different from
what normal looks like," not just "different from the dataset average,"
which would be skewed by the anomalies themselves.
"""

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, get_X

FEATURE_LABELS = {
    "is_cold_start": "no prior history for this entity",
    "log_prior_session_count": "unusual account activity volume",
    "new_resource_flag": "access to a resource never used before",
    "new_device_flag": "new/unrecognized device fingerprint",
    "hour_zscore": "highly unusual time of day",
    "off_hours_flag": "activity outside normal hours",
    "log_time_gap_seconds": "abnormal gap since last activity",
    "log_implied_speed_kmh": "implausible travel speed between locations",
    "duration_zscore": "unusual session duration",
    "auth_failure_flag": "failed authentication attempt",
    "rolling_failure_rate": "elevated recent authentication failure rate",
    "log_ip_recent_failure_count": "many recent failures from this source IP",
    "log_ip_recent_event_count": "unusually high recent activity from this source IP",
    "log_ip_recent_distinct_entities": "this source IP targeting many different accounts",
    "resource_sensitive_flag": "access to a sensitive resource",
}


class ReasonGenerator:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, feats: pd.DataFrame):
        """Fit standardization stats on NORMAL rows only, so deviation is
        measured against genuine baseline behaviour, not a distribution
        already contaminated by the anomalies we're trying to explain."""
        normal = feats[feats["label"] == "normal"] if "label" in feats.columns else feats
        X = get_X(normal)
        self.mean_ = X.mean()
        self.std_ = X.std().replace(0, 1.0)
        return self

    def explain_row(self, feats_row: pd.Series, top_k: int = 3) -> dict:
        X = feats_row[FEATURE_COLUMNS].fillna(0)
        z = ((X - self.mean_) / self.std_).abs()
        top_features = z.sort_values(ascending=False).head(top_k)

        factors = []
        for feat_name, magnitude in top_features.items():
            if magnitude < 0.5:   # not actually deviating meaningfully
                continue
            factors.append({
                "feature": feat_name,
                "description": FEATURE_LABELS.get(feat_name, feat_name),
                "deviation_magnitude": round(float(magnitude), 2),
            })

        if factors:
            summary = "Flagged due to " + " + ".join(f["description"] for f in factors) + "."
        else:
            summary = "Flagged as a statistical outlier; no single dominant factor."

        return {"summary": summary, "top_factors": factors}

    def explain_batch(self, feats: pd.DataFrame, top_k: int = 3) -> list:
        return [self.explain_row(row, top_k) for _, row in feats.iterrows()]


class ClassifierSHAPExplainer:
    """
    Real, model-derived explainability for the ATTACK-TYPE classification
    (as opposed to ReasonGenerator above, which explains why something was
    flagged as anomalous AT ALL). Uses shap.TreeExplainer on the trained
    RandomForest - exact, game-theoretically grounded feature attributions
    for tree ensembles, not a heuristic approximation.

    This directly answers a sharper question than the z-score reason does:
    not just "what's unusual about this session" but "what specifically
    made the model call it credential_stuffing rather than, say,
    lateral_movement" - the SHAP values are computed per PREDICTED CLASS,
    so they explain the classification decision itself.
    """

    def __init__(self, classifier):
        import shap
        self.classes_ = list(classifier.model.classes_)
        self.explainer = shap.TreeExplainer(classifier.model)

    def explain_batch(self, feats: pd.DataFrame, predicted_types: pd.Series, top_k: int = 3) -> list:
        X = get_X(feats)
        sv = self.explainer(X)  # sv.values shape: (n_rows, n_features, n_classes)

        results = []
        for i, pred_type in enumerate(predicted_types):
            if pred_type not in self.classes_:
                results.append({"summary": "", "top_factors": []})
                continue
            class_idx = self.classes_.index(pred_type)
            contributions = sv.values[i, :, class_idx]

            order = np.argsort(-np.abs(contributions))[:top_k]
            factors = []
            for idx in order:
                feat_name = FEATURE_COLUMNS[idx]
                contrib = float(contributions[idx])
                if abs(contrib) < 0.001:
                    continue
                factors.append({
                    "feature": feat_name,
                    "description": FEATURE_LABELS.get(feat_name, feat_name),
                    "shap_contribution": round(contrib, 4),
                    "direction": "supports" if contrib > 0 else "argues against",
                })
            if factors:
                top_supporting = [f for f in factors if f["direction"] == "supports"]
                if top_supporting:
                    summary = (f"Classified as {pred_type.replace('_', ' ')} primarily due to " +
                               " + ".join(f["description"] for f in top_supporting) + ".")
                else:
                    summary = f"Classified as {pred_type.replace('_', ' ')} (low-confidence, no single dominant factor)."
            else:
                summary = f"Classified as {pred_type.replace('_', ' ')}."
            results.append({"summary": summary, "top_factors": factors})
        return results
