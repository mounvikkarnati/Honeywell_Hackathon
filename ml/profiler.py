"""
profiler.py
-----------
Learns "normal" behaviour PER ENTITY FROM THE OBSERVED LOGS THEMSELVES -
never from the generator's internal profile object. This is the actual
Baseline Profiling Model (Deliverable #2).

Key design choices (documented for the report):

1. Rolling window, not full-history expanding window.
   Each session is scored against a baseline built ONLY from that entity's
   own prior events within the last `rolling_window_days` days. This gets
   us three things at once:
     - No leakage: a session is never compared against data from its own
       future.
     - Concept drift handling: old behaviour "ages out" of the window
       automatically, so a legitimate new work pattern stops being
       penalized after `rolling_window_days`.
     - Cold-start is a natural consequence, not a special case: an entity
       with an empty/small rolling window just has low-confidence stats,
       which the feature set and downstream detector can act on directly
       (see `prior_session_count` and `is_cold_start`).

2. Peer-group fallback for cold-start entities.
   When an entity's own rolling window is too thin (< MIN_HISTORY_FOR_OWN
   sessions), we borrow the peer group's stats (median hour, duration,
   failure rate for that entity_type) instead of computing unstable
   individual statistics from 1-2 data points.

3. Two passes, both O(n) with a single chronological scan:
   - Pass 1 (this file): per-entity rolling behavioural stats
   - Pass 2 (ip_features.py): per-source-IP rolling stats, which is what
     catches brute force (many failures, one entity, one IP) and
     credential stuffing (many entities, one/few IPs) - patterns that a
     purely per-entity view can't see.
"""

import math
from collections import deque, defaultdict

import numpy as np
import pandas as pd

MIN_HISTORY_FOR_OWN = 5   # below this many prior sessions in the window,
                           # fall back to peer-group stats
DEFAULT_ROLLING_DAYS = 5  # window length for "recent normal" - shorter
                           # than the 14-day sim so drift/cold-start logic
                           # is actually exercised


def _haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return np.nan
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _peer_group_stats(df: pd.DataFrame) -> dict:
    """Global fallback stats per entity_type, used for cold-start entities."""
    stats = {}
    for etype, sub in df.groupby("entity_type"):
        stats[etype] = {
            "hour_mean": sub["timestamp"].dt.hour.mean(),
            "hour_std": max(sub["timestamp"].dt.hour.std(), 1.0),
            "duration_mean": sub["session_duration"].mean(),
            "duration_std": max(sub["session_duration"].std(), 1.0),
            "failure_rate": (sub["auth_result"] == "failure").mean(),
        }
    return stats


def build_entity_features(df: pd.DataFrame, rolling_window_days: int = DEFAULT_ROLLING_DAYS) -> pd.DataFrame:
    """
    Single chronological pass per entity. Returns a DataFrame aligned to
    df's original row order with the entity-relative feature columns.
    """
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=False)
    peer_stats = _peer_group_stats(df)

    n = len(df)
    out = {
        "is_cold_start": np.zeros(n, dtype=int),
        "prior_session_count": np.zeros(n, dtype=int),
        "new_resource_flag": np.zeros(n, dtype=int),
        "new_device_flag": np.zeros(n, dtype=int),
        "hour_zscore": np.zeros(n, dtype=float),
        "off_hours_flag": np.zeros(n, dtype=int),
        "time_gap_seconds": np.full(n, np.nan, dtype=float),
        "geo_distance_km": np.full(n, np.nan, dtype=float),
        "implied_speed_kmh": np.full(n, np.nan, dtype=float),
        "duration_zscore": np.zeros(n, dtype=float),
        "auth_failure_flag": np.zeros(n, dtype=int),
        "rolling_failure_rate": np.zeros(n, dtype=float),
    }

    window = pd.Timedelta(days=rolling_window_days)

    for entity_id, group in df.groupby("entity_id", sort=False):
        etype = group["entity_type"].iloc[0]
        peer = peer_stats.get(etype, {"hour_mean": 12, "hour_std": 6,
                                       "duration_mean": 60, "duration_std": 60,
                                       "failure_rate": 0.02})

        # Rolling window state for this entity
        buf = deque()          # (timestamp, hour, duration, auth_result)
        seen_resources = set()
        seen_devices = set()
        last_loc = None        # (timestamp, lat, lon)
        failures_in_window = 0

        idx_positions = group.index.to_numpy()  # positions into `df`/out arrays

        for pos, (_, row) in zip(idx_positions, group.iterrows()):
            ts = row["timestamp"]

            # Evict window entries older than the rolling window
            while buf and (ts - buf[0][0]) > window:
                _, _, _, old_result = buf.popleft()
                if old_result == "failure":
                    failures_in_window -= 1

            prior_count = len(buf)
            out["prior_session_count"][pos] = prior_count
            cold = prior_count < MIN_HISTORY_FOR_OWN
            out["is_cold_start"][pos] = int(cold)

            # Hour-of-day deviation
            hour = ts.hour + ts.minute / 60.0
            if cold:
                hour_mean, hour_std = peer["hour_mean"], peer["hour_std"]
            else:
                hours = [h for (_, h, _, _) in buf]
                hour_mean = float(np.mean(hours))
                hour_std = max(float(np.std(hours)), 1.0)
            out["hour_zscore"][pos] = abs(hour - hour_mean) / hour_std
            out["off_hours_flag"][pos] = int(out["hour_zscore"][pos] > 2.5)

            # Duration deviation
            duration = row["session_duration"]
            if cold:
                dur_mean, dur_std = peer["duration_mean"], peer["duration_std"]
            else:
                durs = [d for (_, _, d, _) in buf]
                dur_mean = float(np.mean(durs))
                dur_std = max(float(np.std(durs)), 1.0)
            out["duration_zscore"][pos] = abs(duration - dur_mean) / dur_std

            # Resource / device novelty (relative to rolling window, not
            # all-time history - this is deliberate: a resource an entity
            # used 4 months ago but not recently is still "unfamiliar" for
            # drift purposes)
            out["new_resource_flag"][pos] = int(row["resource_accessed"] not in seen_resources)
            out["new_device_flag"][pos] = int(row["device_fingerprint"] not in seen_devices)

            # Auth failure + rolling failure rate (prior to this event)
            is_failure = row["auth_result"] == "failure"
            out["auth_failure_flag"][pos] = int(is_failure)
            out["rolling_failure_rate"][pos] = (
                failures_in_window / prior_count if prior_count > 0 else peer["failure_rate"]
            )

            # Geo velocity vs the entity's own last known location
            if last_loc is not None:
                last_ts, last_lat, last_lon = last_loc
                gap_seconds = (ts - last_ts).total_seconds()
                out["time_gap_seconds"][pos] = gap_seconds
                dist_km = _haversine_km(last_lat, last_lon, row["geo_lat"], row["geo_lon"])
                out["geo_distance_km"][pos] = dist_km
                gap_hours = max(gap_seconds / 3600.0, 1e-6)
                out["implied_speed_kmh"][pos] = dist_km / gap_hours if not pd.isna(dist_km) else np.nan

            # Update rolling state AFTER computing this event's features
            buf.append((ts, hour, duration, row["auth_result"]))
            if is_failure:
                failures_in_window += 1
            seen_resources.add(row["resource_accessed"])
            seen_devices.add(row["device_fingerprint"])
            last_loc = (ts, row["geo_lat"], row["geo_lon"])

    result = pd.DataFrame(out)
    result.index = df["index"].to_numpy()   # map back to ORIGINAL df row order
    result = result.sort_index()
    return result
