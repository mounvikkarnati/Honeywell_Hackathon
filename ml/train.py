"""
train.py
--------
Orchestrates Phase 3 end-to-end:
  1. Load pre-built features (or build them if missing) + ground truth
  2. Chronological train/test split (NOT random - a random split would
     let future information leak into "training" via the rolling-window
     features' construction; a temporal split matches how this would
     actually be deployed: train on history, evaluate on what comes next)
  3. Fit the Isolation Forest detector (unsupervised) on the train split
  4. Fit the attack-type classifier (supervised, non-'normal' rows only)
     on the train split
  5. Score the test split with both models
  6. Run every metric + plot through the evaluation/ module - nothing
     here hand-computes a metric
  7. Save models (joblib) + evaluation artifacts + summary.json (the
     input the Phase 6 report generator will consume)

Usage:
    python3 -m ml.train --outdir ./output --eval-outdir ./output/evaluation
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from .features import build_feature_matrix, FEATURE_COLUMNS
from .detector import AnomalyDetector
from .classifier import AttackTypeClassifier
from evaluation.metrics import (
    binary_labels_from_ground_truth, binary_detection_metrics,
    classification_report_multiclass,
)
from evaluation.confusion import binary_confusion, multiclass_confusion
from evaluation.plots import generate_all_plots


def load_or_build_features(outdir: str) -> pd.DataFrame:
    features_path = os.path.join(outdir, "features.csv")
    logs_path = os.path.join(outdir, "access_logs.csv")

    df = pd.read_csv(logs_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")

    if os.path.exists(features_path):
        cached = pd.read_csv(features_path)
        # Validate the cache actually corresponds to the CURRENT access_logs.csv
        # before trusting it - a stale cache from a previous/different data
        # generation run is exactly what causes NaN labels downstream.
        same_row_count = len(cached) == len(df)
        same_session_ids = same_row_count and set(cached["session_id"]) == set(df["session_id"])
        if same_row_count and same_session_ids:
            print(f"Loading cached features from {features_path} (validated against current access_logs.csv)...")
            cached["timestamp"] = pd.to_datetime(cached["timestamp"], format="ISO8601")
            return cached
        else:
            print(f"Cached {features_path} does NOT match current access_logs.csv "
                  f"(cached rows={len(cached):,} vs current rows={len(df):,}) - rebuilding features...")

    print("Building features from access_logs.csv...")
    feats = build_feature_matrix(df)
    feats.to_csv(features_path, index=False)
    return feats


def temporal_split(feats: pd.DataFrame, train_fraction: float = 0.7):
    feats_sorted = feats.sort_values("timestamp")
    cutoff_idx = int(len(feats_sorted) * train_fraction)
    cutoff_ts = feats_sorted["timestamp"].iloc[cutoff_idx]
    train_mask = feats["timestamp"] < cutoff_ts
    return feats[train_mask].copy(), feats[~train_mask].copy(), cutoff_ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--eval-outdir", type=str, default="./output/evaluation")
    parser.add_argument("--models-outdir", type=str, default="./models")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--alert-budget", type=float, default=0.01,
                         help="Top fraction of events an analyst can realistically review")
    args = parser.parse_args()

    os.makedirs(args.eval_outdir, exist_ok=True)
    os.makedirs(args.models_outdir, exist_ok=True)

    print("[1/6] Loading features + ground truth...")
    feats = load_or_build_features(args.outdir)
    labels = pd.read_csv(os.path.join(args.outdir, "labels.csv"))
    feats = feats.merge(labels[["session_id", "label"]], on="session_id", how="left")

    n_unmatched = feats["label"].isna().sum()
    if n_unmatched > 0:
        raise RuntimeError(
            f"\n\n*** DATA MISMATCH DETECTED ***\n"
            f"{n_unmatched:,} of {len(feats):,} rows in output/features.csv have no matching "
            f"session_id in output/labels.csv.\n"
            f"This means features.csv was built from a DIFFERENT data-generation run than the "
            f"current output/access_logs.csv + labels.csv (e.g. the dataset was regenerated after "
            f"features.csv was cached, or a stale features.csv/labels.csv pair was mixed together).\n\n"
            f"Fix: delete the stale cache and rebuild both from the CURRENT access_logs.csv:\n"
            f"    rm -f {os.path.join(args.outdir, 'features.csv')}\n"
            f"    python3 -m ml.train --outdir {args.outdir} --eval-outdir {args.eval_outdir} "
            f"--models-outdir {args.models_outdir}\n"
        )
    print(f"       -> {len(feats):,} rows, {feats['label'].nunique()} label classes")

    print("[2/6] Temporal train/test split...")
    train_feats, test_feats, cutoff_ts = temporal_split(feats, args.train_fraction)
    print(f"       -> train: {len(train_feats):,} rows (before {cutoff_ts})")
    print(f"       -> test:  {len(test_feats):,} rows (from {cutoff_ts} onward)")
    print(f"       -> train anomaly rate: {(train_feats['label']!='normal').mean()*100:.2f}%")
    print(f"       -> test anomaly rate:  {(test_feats['label']!='normal').mean()*100:.2f}%")

    print("[3/6] Fitting Isolation Forest detector (unsupervised)...")
    detector = AnomalyDetector(contamination=args.contamination)
    detector.fit(train_feats)

    print("[4/6] Fitting attack-type classifier on labeled anomalies (train split)...")
    train_anomalies = train_feats[train_feats["label"] != "normal"]
    classifier = AttackTypeClassifier()
    classifier.fit(train_anomalies, train_anomalies["label"])
    print(f"       -> trained on {len(train_anomalies):,} labeled anomalous rows, "
          f"classes: {list(classifier.classes_)}")

    print("[5/6] Scoring test split + computing metrics...")
    y_true = binary_labels_from_ground_truth(test_feats["label"])
    y_score = detector.score(test_feats).to_numpy()
    y_pred = (detector.predict_outlier(test_feats).to_numpy() == -1).astype(int)

    detection_metrics = binary_detection_metrics(
        y_true, y_score, y_pred, alert_budget_fractions=(0.01, 0.02, 0.05))
    binary_cm = binary_confusion(y_true, y_pred)

    # Classification: oracle setting - evaluate on the TRUE anomalies in
    # the test split (measures "if we correctly flag it, do we correctly
    # type it", decoupled from detection recall)
    test_anomalies = test_feats[test_feats["label"] != "normal"].copy()
    pred_types = classifier.predict(test_anomalies)
    classification_metrics = classification_report_multiclass(test_anomalies["label"], pred_types)
    multiclass_cm = multiclass_confusion(test_anomalies["label"], pred_types)

    print("[6/6] Generating plots + saving artifacts...")
    plot_paths = generate_all_plots(y_true, y_score, binary_cm, args.eval_outdir, multiclass_cm)

    joblib.dump(detector, os.path.join(args.models_outdir, "detector.joblib"))
    joblib.dump(classifier, os.path.join(args.models_outdir, "classifier.joblib"))

    summary = {
        "train_rows": int(len(train_feats)),
        "test_rows": int(len(test_feats)),
        "train_test_cutoff_timestamp": str(cutoff_ts),
        "contamination_setting": args.contamination,
        "detection_metrics": detection_metrics,
        "binary_confusion_matrix": binary_cm,
        "classification_metrics": classification_metrics,
        "multiclass_confusion_matrix": multiclass_cm,
        "classifier_feature_importances": classifier.feature_importances().to_dict(),
        "plot_paths": plot_paths,
    }
    summary_path = os.path.join(args.eval_outdir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== Detection metrics (test split) ===")
    print(json.dumps(detection_metrics, indent=2, default=str))
    print(f"\nSaved: {summary_path}")
    print(f"Saved models to: {args.models_outdir}/")
    print(f"Saved plots to: {args.eval_outdir}/")


if __name__ == "__main__":
    main()
