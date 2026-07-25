"""
run_pipeline.py
----------------
Single entry point that runs the full Phase 1-4 pipeline in the ONE
correct order, every time. This exists because we hit real, repeated
bugs from running these steps out of order or partially during
development:

  - ml.train needs output/access_logs.csv + labels.csv (Phase 1)
  - backend.ingest loads sessions/entities into the DB (Phase 2) and
    WIPES the alerts table when --reset is used
  - ml.populate_alerts needs the DB to already have sessions loaded
    (Phase 2) AND trained models to exist (Phase 3), and writes alerts
    that the dashboard (Phase 4) depends on

Get this order wrong and you get alerts referencing sessions that don't
exist, or a dashboard reading an empty alerts table - both of which
happened during this build and cost real debugging time. This script
makes "run everything correctly" a single command instead of five you
have to remember the order of.

Usage:
    python3 -m run_pipeline                       # full run, fresh data
    python3 -m run_pipeline --skip-generate        # reuse existing output/access_logs.csv
    python3 -m run_pipeline --skip-generate --skip-train   # just re-ingest + re-populate alerts
"""

import argparse
import os
import subprocess
import sys
import time


def run_step(description: str, cmd: list, cwd: str = "."):
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    print(f"  $ {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n❌ FAILED after {elapsed:.1f}s: {description}")
        print(f"   Pipeline stopped - fix the error above before continuing.")
        sys.exit(1)
    print(f"\n✅ Done in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="./output")
    parser.add_argument("--eval-outdir", default="./output/evaluation")
    parser.add_argument("--models-dir", default="./models")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--users", type=int, default=300)
    parser.add_argument("--service-accounts", type=int, default=15)
    parser.add_argument("--edge-devices", type=int, default=15)
    parser.add_argument("--anomaly-rate", type=float, default=0.02)
    parser.add_argument("--skip-generate", action="store_true",
                         help="Reuse existing output/access_logs.csv instead of regenerating")
    parser.add_argument("--skip-train", action="store_true",
                         help="Reuse existing trained models instead of retraining")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_generate:
        run_step("Phase 1: Generate synthetic data", [
            py, "-m", "generator.main",
            "--days", str(args.days), "--users", str(args.users),
            "--service-accounts", str(args.service_accounts),
            "--edge-devices", str(args.edge_devices),
            "--anomaly-rate", str(args.anomaly_rate),
            "--outdir", args.outdir,
        ])
    else:
        if not os.path.exists(os.path.join(args.outdir, "access_logs.csv")):
            print(f"❌ --skip-generate given but {args.outdir}/access_logs.csv doesn't exist. Aborting.")
            sys.exit(1)
        print(f"\n(skipping Phase 1 - reusing existing {args.outdir}/access_logs.csv)")

    # Phase 2 ingest MUST run before populate_alerts, and it wipes alerts -
    # so it always runs here regardless of skip flags, right before Phase 3/4.
    run_step("Phase 2: Ingest into database (sessions + entities)", [
        py, "-m", "backend.ingest", "--outdir", args.outdir, "--reset",
    ])

    if not args.skip_train:
        run_step("Phase 3: Train models + evaluate", [
            py, "-m", "ml.train",
            "--outdir", args.outdir, "--eval-outdir", args.eval_outdir,
            "--models-outdir", args.models_dir,
        ])
    else:
        if not os.path.exists(os.path.join(args.models_dir, "detector.joblib")):
            print(f"❌ --skip-train given but {args.models_dir}/detector.joblib doesn't exist. Aborting.")
            sys.exit(1)
        print(f"\n(skipping Phase 3 - reusing existing models in {args.models_dir}/)")

    run_step("Phase 4/5: Score sessions + populate alerts (with SHAP explanations)", [
        py, "-m", "ml.populate_alerts", "--outdir", args.outdir, "--models-dir", args.models_dir,
    ])

    print(f"\n{'='*70}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*70}")
    print("  Start the server with:")
    print("    uvicorn backend.app:app --reload --reload-dir backend --reload-dir generator --port 8000")
    print("  Then open http://localhost:8000/ for the dashboard.")


if __name__ == "__main__":
    main()
