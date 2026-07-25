"""
ip_features.py
--------------
Pass 2 of feature engineering: rolling stats grouped by source_ip instead
of by entity. This is what actually catches the two attack patterns a
purely per-entity view is blind to:

  - brute_force:          many FAILURES, one entity, one IP, short window
  - credential_stuffing:  many DISTINCT ENTITIES, one/few IPs, short window

Both show up as spikes in "activity from this IP in the last few minutes",
just with a different entity-diversity signature - which is exactly the
two features computed here.
"""

from collections import deque

import numpy as np
import pandas as pd

IP_WINDOW_MINUTES = 10


def build_ip_features(df: pd.DataFrame, window_minutes: int = IP_WINDOW_MINUTES) -> pd.DataFrame:
    df = df.sort_values(["source_ip", "timestamp"]).reset_index(drop=False)
    window = pd.Timedelta(minutes=window_minutes)

    n = len(df)
    out = {
        "ip_recent_failure_count": np.zeros(n, dtype=int),
        "ip_recent_event_count": np.zeros(n, dtype=int),
        "ip_recent_distinct_entities": np.zeros(n, dtype=int),
    }

    for source_ip, group in df.groupby("source_ip", sort=False):
        buf = deque()  # (timestamp, entity_id, auth_result)
        entity_counts = {}   # entity_id -> count currently in window
        failures_in_window = 0

        for pos, (_, row) in zip(group.index.to_numpy(), group.iterrows()):
            ts = row["timestamp"]

            while buf and (ts - buf[0][0]) > window:
                _, old_entity, old_result = buf.popleft()
                entity_counts[old_entity] -= 1
                if entity_counts[old_entity] == 0:
                    del entity_counts[old_entity]
                if old_result == "failure":
                    failures_in_window -= 1

            out["ip_recent_event_count"][pos] = len(buf)
            out["ip_recent_failure_count"][pos] = failures_in_window
            out["ip_recent_distinct_entities"][pos] = len(entity_counts)

            buf.append((ts, row["entity_id"], row["auth_result"]))
            entity_counts[row["entity_id"]] = entity_counts.get(row["entity_id"], 0) + 1
            if row["auth_result"] == "failure":
                failures_in_window += 1

    result = pd.DataFrame(out)
    result.index = df["index"].to_numpy()
    result = result.sort_index()
    return result
