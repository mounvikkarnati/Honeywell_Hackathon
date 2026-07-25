"""
stream.py
---------
Simulates a live event feed by replaying stored sessions in timestamp
order, compressed into a short real-time window. This is what makes the
dashboard demo feel like a live SOC feed instead of a static CSV viewer,
without needing any real streaming infrastructure (Kafka etc.) for the
hackathon build.

The compression factor is computed from the dataset's actual time span
and a target demo duration, so it works regardless of how many days of
data the generator produced.
"""

import asyncio
import json
from datetime import datetime

from sqlalchemy import asc

from .database import SessionLocal, AccessSession, Alert

MAX_GAP_SECONDS = 2.0  # cap the wait between events so a long quiet period
                        # (e.g. overnight) doesn't stall the demo


def _row_to_dict(row: AccessSession, alert_lookup: dict) -> dict:
    alert = alert_lookup.get(row.session_id)
    return {
        "session_id": row.session_id,
        "event_id": row.event_id,
        "entity_id": row.entity_id,
        "entity_type": row.entity_type,
        "timestamp": row.timestamp.isoformat(),
        "source_ip": row.source_ip,
        "geo_city": row.geo_city,
        "geo_country": row.geo_country,
        "geo_lat": row.geo_lat,
        "geo_lon": row.geo_lon,
        "resource_accessed": row.resource_accessed,
        "auth_method": row.auth_method,
        "auth_result": row.auth_result,
        "session_duration": row.session_duration,
        "device_os": row.device_os,
        "protocol": row.protocol,
        # Real model output, precomputed by ml/populate_alerts.py - None
        # for sessions the detector didn't flag, so the live feed can
        # visually distinguish "just happened" from "just happened AND
        # the model thinks it's suspicious".
        "risk_score": alert.risk_score if alert else None,
        "predicted_type": alert.predicted_type if alert else None,
    }


async def replay_sessions(websocket, demo_duration_seconds: float = 120.0,
                           limit: int = 20000):
    """Streams stored sessions over the given websocket, oldest first,
    with inter-event gaps compressed to fit demo_duration_seconds total."""
    db = SessionLocal()
    try:
        rows = (
            db.query(AccessSession)
            .order_by(asc(AccessSession.timestamp))
            .limit(limit)
            .all()
        )
        if len(rows) < 2:
            await websocket.send_json({"error": "not enough sessions to stream"})
            return

        alert_lookup = {a.session_id: a for a in db.query(Alert).all()}

        t_start = rows[0].timestamp
        t_end = rows[-1].timestamp
        real_span = max((t_end - t_start).total_seconds(), 1.0)
        scale = real_span / demo_duration_seconds  # e.g. 14 days -> 120s means scale ~= 10080

        prev_ts = rows[0].timestamp
        for row in rows:
            gap = (row.timestamp - prev_ts).total_seconds() / scale
            gap = max(0.0, min(gap, MAX_GAP_SECONDS))
            if gap > 0:
                await asyncio.sleep(gap)
            await websocket.send_json(_row_to_dict(row, alert_lookup))
            prev_ts = row.timestamp
    finally:
        db.close()
