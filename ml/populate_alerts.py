"""
populate_alerts.py
-------------------
Runs the trained detector + classifier + explainer over the dataset and
writes results into the `alerts` DB table. This is what turns the Phase 2
`/alerts` stub into a real, populated endpoint the dashboard can show.

Only sessions the detector actually flags as outliers become alert rows -
an "alert queue" that contained every session would defeat the purpose of
an alert queue (this mirrors the "realistic analyst alert budget"
evaluation criterion: we're not dumping all 109k events on the analyst).

Usage:
    python3 -m ml.populate_alerts --outdir ./output --models-dir ./models
"""

import argparse
import json
import os

import joblib
import pandas as pd
from sqlalchemy.orm import Session as OrmSession

from backend.database import SessionLocal, Alert, init_db
from .features import build_feature_matrix
from .explain import ReasonGenerator, ClassifierSHAPExplainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--models-dir", type=str, default="./models")
    args = parser.parse_args()

    print("[1/5] Loading trained models...")
    detector = joblib.load(os.path.join(args.models_dir, "detector.joblib"))
    classifier = joblib.load(os.path.join(args.models_dir, "classifier.joblib"))

    print("[2/5] Loading features (rebuilding if needed)...")
    features_path = os.path.join(args.outdir, "features.csv")
    logs_path = os.path.join(args.outdir, "access_logs.csv")
    df = pd.read_csv(logs_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")

    if os.path.exists(features_path):
        cached = pd.read_csv(features_path)
        if len(cached) == len(df) and set(cached["session_id"]) == set(df["session_id"]):
            feats = cached
            feats["timestamp"] = pd.to_datetime(feats["timestamp"], format="ISO8601")
        else:
            feats = build_feature_matrix(df)
            feats.to_csv(features_path, index=False)
    else:
        feats = build_feature_matrix(df)
        feats.to_csv(features_path, index=False)

    labels_path = os.path.join(args.outdir, "labels.csv")
    labels = pd.read_csv(labels_path)
    feats_with_label = feats.merge(labels[["session_id", "label"]], on="session_id", how="left")

    print("[3/5] Fitting reason generator on normal-row statistics...")
    reasoner = ReasonGenerator().fit(feats_with_label)

    print("[4/5] Scoring all sessions...")
    risk_scores_all = detector.score(feats)   # scored once, over the FULL dataset,
                                                # so the 0-100 scale is consistent
                                                # across every alert (avoids the
                                                # batch-relative min-max distortion
                                                # documented in ml/METHODOLOGY.md)
    is_outlier = (detector.predict_outlier(feats) == -1)

    # Supplementary rule-based flag: a new device fingerprint appearing on an
    # entity that ALREADY has established history is a near-deterministic
    # device-spoofing signature by construction (see generator/anomalies.py).
    # The Isolation Forest alone misses ~all device_spoofing instances at the
    # current contamination setting (verified: 0/25 true instances flagged in
    # testing) because it's tuned to the overall ~2% base rate, which is
    # dominated by louder, more common patterns like brute force. Real SOC
    # tooling commonly layers targeted rules like this on top of a general
    # statistical model for exactly this reason - documented explicitly here
    # rather than silently patched.
    rule_based_flag = (feats["new_device_flag"] == 1) & (feats["is_cold_start"] == 0)
    combined_flag = is_outlier | rule_based_flag

    flagged = feats[combined_flag].copy()
    flagged["risk_score"] = risk_scores_all[combined_flag]
    flagged["flag_source"] = [
        "statistical" if o else "rule_based_device_check"
        for o in is_outlier[combined_flag]
    ]
    print(f"       -> {is_outlier.sum():,} flagged by Isolation Forest, "
          f"{(rule_based_flag & ~is_outlier).sum():,} additional flagged by the "
          f"device-fingerprint rule, {len(flagged):,} total "
          f"({len(flagged)/len(feats)*100:.2f}% of all sessions)")

    print("       -> classifying attack type for flagged sessions...")
    type_preds = classifier.predict_proba_top(flagged)
    flagged = flagged.merge(type_preds, left_index=True, right_index=True)

    print("       -> generating detection-level explanations for flagged sessions...")
    reasons = reasoner.explain_batch(flagged)

    print("       -> generating SHAP-based classification explanations...")
    shap_explainer = ClassifierSHAPExplainer(classifier)
    classification_reasons = shap_explainer.explain_batch(flagged, flagged["predicted_type"])

    print("[5/5] Writing alerts to database...")
    init_db()
    db: OrmSession = SessionLocal()
    try:
        db.query(Alert).delete()  # idempotent re-run
        db.commit()

        rows = []
        for (_, row), detection_reason, classification_reason in zip(
                flagged.iterrows(), reasons, classification_reasons):
            combined_reason = {
                "summary": detection_reason["summary"],
                "top_factors": detection_reason["top_factors"],
                "flag_source": row["flag_source"],
                "classification": classification_reason,   # SHAP-based "why this specific type"
            }
            if row["flag_source"] == "rule_based_device_check":
                combined_reason["summary"] = (
                    "Flagged by device-fingerprint rule (new device on an "
                    "established account) - " + detection_reason["summary"][0].lower()
                    + detection_reason["summary"][1:]
                )
            rows.append(Alert(
                session_id=row["session_id"],
                entity_id=row["entity_id"],
                risk_score=float(row["risk_score"]),
                predicted_type=row["predicted_type"],
                reason=combined_reason,
            ))
        db.bulk_save_objects(rows)
        db.commit()
        print(f"       -> wrote {len(rows):,} alerts")
    finally:
        db.close()

    print("\nDone. /alerts is now populated with real model output.")


if __name__ == "__main__":
    main()
