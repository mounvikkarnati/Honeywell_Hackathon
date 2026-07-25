"""
features.py
-----------
Assembles the final feature matrix fed to the detector/classifier by
combining:
  - entity-relative rolling features (profiler.py)
  - IP-relative rolling features (ip_features.py)
  - a resource-sensitivity flag (domain knowledge - which resources are
    sensitive is known upfront to any SOC, this is not label leakage)

Skewed count/velocity features are log1p-transformed and clipped so a
handful of extreme outliers (e.g. a 6000 km/h impossible-travel speed)
don't dominate the Isolation Forest's isolation splits disproportionately.
"""

import numpy as np
import pandas as pd

from generator.profiles import SENSITIVE_RESOURCES
from .profiler import build_entity_features
from .ip_features import build_ip_features

FEATURE_COLUMNS = [
    "is_cold_start",
    "log_prior_session_count",
    "new_resource_flag",
    "new_device_flag",
    "hour_zscore",
    "off_hours_flag",
    "log_time_gap_seconds",
    "log_implied_speed_kmh",
    "duration_zscore",
    "auth_failure_flag",
    "rolling_failure_rate",
    "log_ip_recent_failure_count",
    "log_ip_recent_event_count",
    "log_ip_recent_distinct_entities",
    "resource_sensitive_flag",
]


def _safe_log1p(series, clip_max=None):
    s = series.fillna(0).clip(lower=0)
    if clip_max is not None:
        s = s.clip(upper=clip_max)
    return np.log1p(s)


def build_feature_matrix(df: pd.DataFrame, rolling_window_days: int = 5) -> pd.DataFrame:
    """
    df must contain the raw access-log columns (as produced by the
    generator / read from access_logs.csv) with 'timestamp' already
    parsed to datetime. Returns a DataFrame indexed like df with all
    FEATURE_COLUMNS plus passthrough identifier columns for joining
    back to session_id / entity_id / labels later.
    """
    df = df.copy()
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")

    entity_feats = build_entity_features(df, rolling_window_days=rolling_window_days)
    ip_feats = build_ip_features(df)

    feats = pd.concat([entity_feats, ip_feats], axis=1)

    feats["log_prior_session_count"] = _safe_log1p(feats["prior_session_count"])
    feats["log_time_gap_seconds"] = _safe_log1p(feats["time_gap_seconds"], clip_max=None)
    feats["log_implied_speed_kmh"] = _safe_log1p(feats["implied_speed_kmh"], clip_max=20000)
    feats["log_ip_recent_failure_count"] = _safe_log1p(feats["ip_recent_failure_count"])
    feats["log_ip_recent_event_count"] = _safe_log1p(feats["ip_recent_event_count"])
    feats["log_ip_recent_distinct_entities"] = _safe_log1p(feats["ip_recent_distinct_entities"])

    feats["resource_sensitive_flag"] = df["resource_accessed"].isin(SENSITIVE_RESOURCES).astype(int)

    # passthrough identifiers, useful for joining with labels / explainability
    for col in ["session_id", "event_id", "entity_id", "entity_type", "timestamp",
                "resource_accessed", "source_ip", "auth_result", "device_fingerprint"]:
        feats[col] = df[col].values

    return feats


def get_X(feats: pd.DataFrame) -> pd.DataFrame:
    """Extract just the numeric model-input columns, in stable order."""
    X = feats[FEATURE_COLUMNS].fillna(0)
    return X
