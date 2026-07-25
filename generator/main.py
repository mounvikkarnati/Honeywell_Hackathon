"""
main.py
-------
Orchestrates the full synthetic data generation pipeline:

  1. Build per-entity behavioural profiles (profiles.py)
  2. Generate benign baseline sessions for every entity across the sim
     window (baseline.py)
  3. Inject anomaly episodes at controlled rates (anomalies.py), targeting
     an overall anomaly rate of ~0.5-3% of total sessions
  4. Shuffle everything into chronological-ish order, assign a session_id
  5. Write:
       - access_logs.csv     (all fields EXCEPT label -> this is what the
                               detection model sees / what "inference" data
                               looks like)
       - access_logs_full.csv (all fields INCLUDING label, for training /
                               offline evaluation)
       - labels.csv           (event_id -> label only, the ground truth kept
                               separate as required by the schema note:
                               "hidden at inference")
       - entity_profiles.json (serialized profiles, useful for the
                               dashboard's "entity history" view and for
                               documenting behavioural assumptions)

Usage:
    python -m generator.main --days 30 --users 800 --service-accounts 120 \
        --edge-devices 120 --anomaly-rate 0.02 --outdir ../output
"""

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd

from .profiles import build_all_profiles
from .baseline import generate_baseline_sessions
from . import anomalies as anom

ANOMALY_INJECTORS_SINGLE_ENTITY = {
    "brute_force": anom.inject_brute_force,
    "impossible_travel": anom.inject_impossible_travel,
    "lateral_movement": anom.inject_lateral_movement,
    "device_spoofing": anom.inject_device_spoofing,
    "low_and_slow_exfiltration": anom.inject_low_and_slow_exfiltration,
    "insider_drift": anom.inject_insider_drift,
}
# credential_stuffing is multi-entity, handled separately

# Relative weights for how often each pattern is chosen when injecting
# (insider_drift and low_and_slow are rarer / longer-running episodes;
# brute_force / impossible_travel are quick single episodes)
PATTERN_WEIGHTS = {
    "brute_force": 0.22,
    "impossible_travel": 0.16,
    "lateral_movement": 0.16,
    "device_spoofing": 0.14,
    "low_and_slow_exfiltration": 0.12,
    "insider_drift": 0.10,
    "credential_stuffing": 0.10,
}

# How many days each pattern's episode can span, so anchor selection can
# reserve enough room inside the sim window (see _random_anchor_ts).
PATTERN_SPAN_DAYS = {
    "brute_force": 1,
    "impossible_travel": 1,
    "lateral_movement": 1,
    "device_spoofing": 1,
    "low_and_slow_exfiltration": 14,
    "insider_drift": 20,
    "credential_stuffing": 1,
}


def _random_anchor_ts(sim_start, sim_days, episode_span_days=1):
    """Pick an anchor timestamp that leaves enough room for the episode's
    full span to land inside the simulation window (important for
    multi-day patterns like low_and_slow_exfiltration and insider_drift,
    which would otherwise spill past sim_days and produce a misleading
    'dataset spans 33 days when only 14 were generated' date range)."""
    from datetime import timedelta
    max_offset = max(sim_days - episode_span_days, 0)
    day_offset = random.uniform(0, max_offset) if max_offset > 0 else 0
    hour = random.uniform(0, 23.9)
    return sim_start + timedelta(days=day_offset, hours=hour)


def generate_dataset(sim_days=30, n_users=800, n_service_accounts=120,
                      n_edge_devices=120, anomaly_rate=0.02, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    sim_start = datetime(2026, 6, 1, 0, 0, 0)

    print(f"[1/4] Building profiles for {n_users + n_service_accounts + n_edge_devices} entities...")
    profiles = build_all_profiles(n_users, n_service_accounts, n_edge_devices)

    print(f"[2/4] Generating {sim_days}-day baseline traffic...")
    all_sessions = []
    for profile in profiles.values():
        all_sessions.extend(generate_baseline_sessions(profile, sim_start, sim_days))
    n_baseline = len(all_sessions)
    print(f"       -> {n_baseline:,} benign sessions generated")

    # Target number of anomalous EPISODES such that resulting anomaly
    # events land roughly at anomaly_rate of total sessions.
    target_anomaly_events = int(n_baseline * anomaly_rate / (1 - anomaly_rate))
    print(f"[3/4] Injecting anomalies (targeting ~{target_anomaly_events:,} anomalous events, "
          f"{anomaly_rate*100:.1f}% of total)...")

    patterns = list(PATTERN_WEIGHTS.keys())
    weights = list(PATTERN_WEIGHTS.values())

    injected_events = 0
    episode_log = []  # for the attack taxonomy report
    user_like_profiles = [p for p in profiles.values() if p.entity_type == "user"]
    any_profile_list = list(profiles.values())

    while injected_events < target_anomaly_events:
        pattern = random.choices(patterns, weights=weights, k=1)[0]
        span_days = min(PATTERN_SPAN_DAYS[pattern], sim_days)
        anchor_ts = _random_anchor_ts(sim_start, sim_days, episode_span_days=span_days)

        if pattern == "credential_stuffing":
            new_events = anom.inject_credential_stuffing(profiles, anchor_ts)
        elif pattern == "lateral_movement":
            target = random.choice(any_profile_list)
            new_events = anom.inject_lateral_movement(target, anchor_ts)
        elif pattern == "low_and_slow_exfiltration":
            target = random.choice(any_profile_list)
            new_events = anom.inject_low_and_slow_exfiltration(target, anchor_ts, n_days=span_days)
        elif pattern == "insider_drift":
            target = random.choice(any_profile_list)
            new_events = anom.inject_insider_drift(target, anchor_ts, n_days=span_days)
        else:
            target = random.choice(any_profile_list)
            new_events = ANOMALY_INJECTORS_SINGLE_ENTITY[pattern](target, anchor_ts)

        all_sessions.extend(new_events)
        injected_events += len(new_events)
        episode_log.append({
            "pattern": pattern,
            "n_events": len(new_events),
            "anchor_ts": anchor_ts.isoformat(),
        })

    print(f"       -> {injected_events:,} anomalous events injected across {len(episode_log):,} episodes")

    print("[4/4] Assembling final dataset...")
    df = pd.DataFrame(all_sessions)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "session_id", [f"sess_{i:07d}" for i in range(len(df))])

    return df, profiles, episode_log


def write_outputs(df: pd.DataFrame, profiles: dict, episode_log: list, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    labels_df = df[["session_id", "event_id", "entity_id", "label"]].copy()
    features_df = df.drop(columns=["label"])

    # What the detection model actually sees (labels hidden, as the schema requires)
    features_df.to_csv(os.path.join(outdir, "access_logs.csv"), index=False)
    # Full JSON mirror of the CSV (no label) - useful for API/demo consumption
    features_df.to_json(os.path.join(outdir, "access_logs.json"), orient="records")
    # Small human-readable JSON sample for quick inspection / docs
    features_df.head(200).to_json(os.path.join(outdir, "access_logs_sample.json"),
                                    orient="records", indent=2)

    # Full version with ground truth, for training / offline evaluation only
    df.to_csv(os.path.join(outdir, "access_logs_full.csv"), index=False)

    # Ground truth kept separate
    labels_df.to_csv(os.path.join(outdir, "labels.csv"), index=False)

    # Serialize entity profiles (useful for dashboard entity-history view
    # and for documenting behavioural assumptions)
    profiles_serializable = {
        eid: {
            "entity_type": p.entity_type,
            "home_locations": p.home_locations,
            "active_hours": p.active_hours,
            "resource_pool": p.resource_pool,
            "typical_auth": p.typical_auth,
            "device_fingerprints": p.device_fingerprints,
            "sessions_per_day": p.sessions_per_day,
            "avg_session_duration": p.avg_session_duration,
            "role_sensitivity": p.role_sensitivity,
        }
        for eid, p in profiles.items()
    }
    with open(os.path.join(outdir, "entity_profiles.json"), "w") as f:
        json.dump(profiles_serializable, f, indent=2)

    with open(os.path.join(outdir, "injection_episode_log.json"), "w") as f:
        json.dump(episode_log, f, indent=2)

    # Summary stats for quick sanity-check / report figures
    label_counts = df["label"].value_counts().to_dict()
    summary = {
        "total_sessions": int(len(df)),
        "entity_count": len(profiles),
        "label_distribution": label_counts,
        "anomaly_rate_pct": round(100 * (1 - label_counts.get("normal", 0) / len(df)), 3),
    }
    with open(os.path.join(outdir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--users", type=int, default=300)
    parser.add_argument("--service-accounts", type=int, default=15)
    parser.add_argument("--edge-devices", type=int, default=15)
    parser.add_argument("--anomaly-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="./output")
    args = parser.parse_args()

    df, profiles, episode_log = generate_dataset(
        sim_days=args.days,
        n_users=args.users,
        n_service_accounts=args.service_accounts,
        n_edge_devices=args.edge_devices,
        anomaly_rate=args.anomaly_rate,
        seed=args.seed,
    )
    summary = write_outputs(df, profiles, episode_log, args.outdir)
    print("\n=== Dataset Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
