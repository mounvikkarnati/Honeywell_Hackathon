"""
ingest.py
---------
Loads the synthetic data generator's output (Phase 1) into the database
(Phase 2). This is the seam between the two phases - if this script runs
clean, Phase 1 -> Phase 2 integration is verified.

Usage:
    python3 -m backend.ingest --outdir ../output
"""

import argparse
import json
import os
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session as OrmSession

from .database import init_db, SessionLocal, Entity, AccessSession, GroundTruth


def _parse_ts(s):
    return datetime.fromisoformat(s)


def ingest_entities(db: OrmSession, outdir: str):
    with open(os.path.join(outdir, "entity_profiles.json")) as f:
        profiles = json.load(f)

    for entity_id, p in profiles.items():
        db.merge(Entity(
            entity_id=entity_id,
            entity_type=p["entity_type"],
            home_locations=p["home_locations"],
            active_hours=p["active_hours"],
            resource_pool=p["resource_pool"],
            typical_auth=p["typical_auth"],
            device_fingerprints=p["device_fingerprints"],
            sessions_per_day=p["sessions_per_day"],
            avg_session_duration=p["avg_session_duration"],
            role_sensitivity=p["role_sensitivity"],
        ))
    db.commit()
    print(f"  -> ingested {len(profiles):,} entity profiles")


def ingest_sessions(db: OrmSession, outdir: str, batch_size=5000):
    df = pd.read_csv(os.path.join(outdir, "access_logs.csv"))  # NO label column - feature-visible only
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")

    rows = df.to_dict(orient="records")
    n = len(rows)
    for i in range(0, n, batch_size):
        batch = rows[i:i + batch_size]
        db.bulk_insert_mappings(AccessSession, [
            {**r, "timestamp": r["timestamp"]} for r in batch
        ])
        db.commit()
        print(f"  -> ingested sessions {i + len(batch):,}/{n:,}", end="\r")
    print()

    # first_seen / session_count per entity, used for cold-start detection later
    stats = df.groupby("entity_id").agg(
        first_seen=("timestamp", "min"),
        session_count=("timestamp", "count"),
    ).reset_index()
    for _, row in stats.iterrows():
        entity = db.query(Entity).filter_by(entity_id=row["entity_id"]).first()
        if entity:
            entity.first_seen = row["first_seen"].to_pydatetime()
            entity.session_count = int(row["session_count"])
    db.commit()
    print(f"  -> updated first_seen/session_count for {len(stats):,} entities")


def ingest_ground_truth(db: OrmSession, outdir: str, batch_size=5000):
    df = pd.read_csv(os.path.join(outdir, "labels.csv"))
    rows = df.to_dict(orient="records")
    n = len(rows)
    for i in range(0, n, batch_size):
        batch = rows[i:i + batch_size]
        db.bulk_insert_mappings(GroundTruth, batch)
        db.commit()
        print(f"  -> ingested ground truth {i + len(batch):,}/{n:,}", end="\r")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="./output")
    parser.add_argument("--reset", action="store_true",
                         help="Drop and recreate tables before ingesting")
    args = parser.parse_args()

    if args.reset:
        db_path = "cyberanomaly.db"
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Removed existing {db_path}")

    print("[1/3] Initializing database schema...")
    init_db()

    db = SessionLocal()
    try:
        print("[2/3] Ingesting entity profiles...")
        ingest_entities(db, args.outdir)

        print("[3/3] Ingesting sessions + ground truth...")
        ingest_sessions(db, args.outdir)
        ingest_ground_truth(db, args.outdir)
    finally:
        db.close()

    print("\nIngest complete.")


if __name__ == "__main__":
    main()
