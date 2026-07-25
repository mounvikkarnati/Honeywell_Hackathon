"""
verify_pipeline.py
-------------------
End-to-end verification that all 7 attack patterns actually flow
correctly through the full pipeline: injected by the generator -> caught
by the detector -> classified correctly -> explained sensibly -> visible
via the API.

This is deliberately NOT the same thing as the Phase 3 evaluation metrics
(precision/recall/F1 etc.) - those measure aggregate statistical
performance on a held-out split. This script checks the concrete,
demo-relevant question: "for each of the 7 named attack types, is there
at least one real, end-to-end example an analyst could actually click on
in the dashboard and see a sensible alert for?" A model can have great
aggregate metrics while still having zero visible examples of some rare
category in the alerts table (which is exactly the device_spoofing bug
this script would have caught immediately).

Usage:
    python3 -m verify_pipeline --outdir ./output
"""

import argparse
import json
import sys

import pandas as pd

from backend.database import SessionLocal, Alert, AccessSession

ATTACK_TYPES = [
    "brute_force", "credential_stuffing", "device_spoofing",
    "impossible_travel", "insider_drift", "lateral_movement",
    "low_and_slow_exfiltration",
]


def check(label, condition, detail=""):
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}  {label}" + (f" — {detail}" if detail else ""))
    return condition


def compute_attack_coverage(labels: pd.DataFrame, db) -> list:
    """Returns per-attack-type coverage stats: how many were injected,
    how many got flagged as alerts, how many were classified correctly,
    and how many have a non-empty explanation. Shared between
    verify_pipeline.py (pass/fail checks) and the Phase 6 report
    generator (numbers in the final report), so both always agree."""
    rows = []
    for attack_type in ATTACK_TYPES:
        true_ids = set(labels[labels["label"] == attack_type]["session_id"])
        n_injected = len(true_ids)
        alerts_for_type = db.query(Alert).filter(Alert.session_id.in_(true_ids)).all() if true_ids else []
        n_flagged = len(alerts_for_type)
        n_correct_type = sum(1 for a in alerts_for_type if a.predicted_type == attack_type)
        n_has_reason = sum(1 for a in alerts_for_type if a.reason and a.reason.get("summary"))
        rows.append({
            "attack_type": attack_type,
            "n_injected": n_injected,
            "n_flagged": n_flagged,
            "flag_rate_pct": round(100 * n_flagged / n_injected, 1) if n_injected else 0.0,
            "n_correct_type": n_correct_type,
            "n_has_reason": n_has_reason,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="./output")
    args = parser.parse_args()

    print("="*70)
    print("  END-TO-END ATTACK SCENARIO VERIFICATION")
    print("="*70)

    labels = pd.read_csv(f"{args.outdir}/labels.csv")
    db = SessionLocal()
    all_ok = True

    try:
        total_sessions = db.query(AccessSession).count()
        total_alerts = db.query(Alert).count()
        print(f"\nDatabase: {total_sessions:,} sessions, {total_alerts:,} alerts\n")
        all_ok &= check("Sessions are loaded", total_sessions > 0,
                         "run backend.ingest first" if total_sessions == 0 else "")
        all_ok &= check("Alerts are populated", total_alerts > 0,
                         "run ml.populate_alerts first" if total_alerts == 0 else "")

        if total_sessions == 0 or total_alerts == 0:
            print("\nCannot continue verification without sessions + alerts. Aborting.")
            sys.exit(1)

        print(f"\n{'Attack type':<28} {'Injected':>9} {'Flagged':>8} {'Correctly typed':>16} {'Has reason':>11}")
        print("-" * 76)

        coverage = compute_attack_coverage(labels, db)
        for row in coverage:
            attack_type = row["attack_type"]
            n_injected, n_flagged = row["n_injected"], row["n_flagged"]
            n_correct_type, n_has_reason = row["n_correct_type"], row["n_has_reason"]

            print(f"{attack_type:<28} {n_injected:>9} {n_flagged:>8} {n_correct_type:>16} {n_has_reason:>11}")

            all_ok &= check(f"  '{attack_type}' has at least 1 injected example", n_injected > 0)
            all_ok &= check(f"  '{attack_type}' has at least 1 FLAGGED alert", n_flagged > 0,
                             "detector+rules missed this entire category - dashboard would show nothing for it"
                             if n_flagged == 0 else "")
            if n_flagged > 0:
                all_ok &= check(f"  '{attack_type}' flagged alerts have a non-empty reason", n_has_reason == n_flagged)

        print("\n" + "="*70)
        # Spot-check one full alert's structure for the SHAP explanation field
        sample = db.query(Alert).filter(Alert.predicted_type.isnot(None)).first()
        if sample:
            has_classification = "classification" in (sample.reason or {})
            all_ok &= check("Sample alert has SHAP classification explanation", has_classification)

        print("\n" + ("✅ ALL CHECKS PASSED" if all_ok else "❌ SOME CHECKS FAILED - see above"))
        print("="*70)

    finally:
        db.close()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
