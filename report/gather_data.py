"""
gather_data.py
---------------
Assembles ALL the data the report needs into one dict, pulling live from:
  - output/evaluation/summary.json      (Phase 3 detection + classification metrics)
  - output/dataset_summary.json         (Phase 1 dataset composition)
  - the live database                   (Phase 5's attack-coverage table)

Nothing here is hand-typed or copy-pasted from a previous run - every
number is read fresh, so the generated report can never drift from
what the current models/data actually produced. This is the literal
implementation of "auto-fill: number of attacks, detection rate, false
positives, precision, recall" from the deliverable spec.
"""

import json
import os

import pandas as pd
from sqlalchemy import func

from backend.database import SessionLocal, Alert
from verify_pipeline import compute_attack_coverage


def gather(outdir: str = "./output", eval_outdir: str = "./output/evaluation") -> dict:
    with open(os.path.join(eval_outdir, "summary.json")) as f:
        eval_summary = json.load(f)

    with open(os.path.join(outdir, "dataset_summary.json")) as f:
        dataset_summary = json.load(f)

    labels = pd.read_csv(os.path.join(outdir, "labels.csv"))
    db = SessionLocal()
    try:
        attack_coverage = compute_attack_coverage(labels, db)
        total_alerts = db.query(func.count(Alert.session_id)).scalar()
    finally:
        db.close()

    dm = eval_summary["detection_metrics"]
    cm = eval_summary["classification_metrics"]

    return {
        "dataset": dataset_summary,
        "attack_coverage": attack_coverage,
        "total_alerts_in_queue": total_alerts,
        "detection_metrics": dm,
        "binary_confusion_matrix": eval_summary["binary_confusion_matrix"],
        "classification_metrics": cm,
        "classifier_feature_importances": eval_summary["classifier_feature_importances"],
        "train_test_cutoff": eval_summary["train_test_cutoff_timestamp"],
        "train_rows": eval_summary["train_rows"],
        "test_rows": eval_summary["test_rows"],
        "eval_outdir": eval_outdir,
    }
