"""
app.py
------
Main FastAPI application. Endpoint contract for Phase 2:

  GET  /health
  GET  /entities                     (list, filter by entity_type, paginated)
  GET  /entities/{entity_id}         (single entity profile)
  GET  /entities/{entity_id}/history (session history for that entity)
  GET  /sessions                     (paginated, filterable session log)
  GET  /stats                        (basic dataset stats - entity-visible)
  GET  /admin/ground-truth-stats     (dev/eval-only label distribution)
  GET  /alerts                       (STUB until Phase 3 - always returns
                                       empty with a note; real detector
                                       wires in here without changing the
                                       contract)
  WS   /ws/stream                    (simulated live event feed)

Run with:
    uvicorn backend.app:app --reload --port 8000
"""

import json
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func
from sqlalchemy.orm import Session as OrmSession

from .database import get_db, Entity, AccessSession, GroundTruth, Alert, init_db
from .schemas import (SessionOut, EntityOut, StatsOut, GroundTruthStatsOut,
                       AlertOut, AlertDetailOut, DeviceTrustOut)
from .stream import replay_sessions

app = FastAPI(title="Cyber Anomaly Detection API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hackathon scope; tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()  # safe no-op if tables already exist / already ingested


FRONTEND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dashboard.html")
EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "evaluation")

if os.path.isdir(EVAL_DIR):
    app.mount("/evaluation-assets", StaticFiles(directory=EVAL_DIR), name="evaluation-assets")


@app.get("/evaluation/summary")
def evaluation_summary():
    """Serves ml/train.py's summary.json - the same numbers the report
    generator (Phase 6) will use, so the dashboard's Model Evaluation tab
    and the final report can never show different numbers for the same run."""
    summary_path = os.path.join(EVAL_DIR, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    return {"error": "no evaluation summary found - run ml/train.py first"}


@app.get("/")
def dashboard():
    """Serves the analyst dashboard. Single self-contained HTML file
    (React via CDN, no Node/npm build step) - deliberate choice to avoid
    adding a second toolchain on top of the Python one, after the
    Python/venv setup issues already hit earlier in this build."""
    if os.path.exists(FRONTEND_PATH):
        return FileResponse(FRONTEND_PATH)
    return {"error": "dashboard.html not found - run from project root"}


@app.get("/stats/alerts")
def alert_stats(db: OrmSession = Depends(get_db)):
    """Aggregate alert stats for the dashboard's Stats page: counts by
    predicted attack type and a risk-score histogram."""
    total_alerts = db.query(func.count(Alert.session_id)).scalar()
    by_type = dict(
        db.query(Alert.predicted_type, func.count(Alert.session_id))
        .group_by(Alert.predicted_type)
        .all()
    )
    all_scores = [r[0] for r in db.query(Alert.risk_score).all()]
    bins = list(range(0, 101, 10))
    histogram = {f"{bins[i]}-{bins[i+1]}": 0 for i in range(len(bins) - 1)}
    for s in all_scores:
        idx = min(int(s // 10), 9)
        key = f"{bins[idx]}-{bins[idx+1]}"
        histogram[key] += 1
    return {
        "total_alerts": total_alerts,
        "by_predicted_type": by_type,
        "risk_score_histogram": histogram,
    }


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/entities", response_model=list[EntityOut])
def list_entities(
    entity_type: Optional[str] = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
    db: OrmSession = Depends(get_db),
):
    q = db.query(Entity)
    if entity_type:
        q = q.filter(Entity.entity_type == entity_type)
    return q.order_by(Entity.entity_id).offset(offset).limit(limit).all()


@app.get("/entities/{entity_id}", response_model=EntityOut)
def get_entity(entity_id: str, db: OrmSession = Depends(get_db)):
    entity = db.query(Entity).filter(Entity.entity_id == entity_id).first()
    if not entity:
        return {"error": f"entity {entity_id} not found"}
    return entity


@app.get("/entities/{entity_id}/history", response_model=list[SessionOut])
def get_entity_history(
    entity_id: str,
    limit: int = Query(200, le=2000),
    db: OrmSession = Depends(get_db),
):
    return (
        db.query(AccessSession)
        .filter(AccessSession.entity_id == entity_id)
        .order_by(AccessSession.timestamp.desc())
        .limit(limit)
        .all()
    )


@app.get("/sessions", response_model=list[SessionOut])
def list_sessions(
    entity_id: Optional[str] = None,
    resource_accessed: Optional[str] = None,
    auth_result: Optional[str] = None,
    limit: int = Query(100, le=2000),
    offset: int = 0,
    db: OrmSession = Depends(get_db),
):
    q = db.query(AccessSession)
    if entity_id:
        q = q.filter(AccessSession.entity_id == entity_id)
    if resource_accessed:
        q = q.filter(AccessSession.resource_accessed == resource_accessed)
    if auth_result:
        q = q.filter(AccessSession.auth_result == auth_result)
    return q.order_by(AccessSession.timestamp.desc()).offset(offset).limit(limit).all()


@app.get("/stats", response_model=StatsOut)
def stats(db: OrmSession = Depends(get_db)):
    total_sessions = db.query(func.count(AccessSession.session_id)).scalar()
    total_entities = db.query(func.count(Entity.entity_id)).scalar()

    type_breakdown = dict(
        db.query(Entity.entity_type, func.count(Entity.entity_id))
        .group_by(Entity.entity_type)
        .all()
    )

    min_ts = db.query(func.min(AccessSession.timestamp)).scalar()
    max_ts = db.query(func.max(AccessSession.timestamp)).scalar()

    return StatsOut(
        total_sessions=total_sessions,
        total_entities=total_entities,
        entity_type_breakdown=type_breakdown,
        date_range={
            "start": min_ts.isoformat() if min_ts else None,
            "end": max_ts.isoformat() if max_ts else None,
        },
    )


@app.get("/admin/ground-truth-stats", response_model=GroundTruthStatsOut)
def ground_truth_stats(db: OrmSession = Depends(get_db)):
    """Dev/evaluation-only endpoint. Reads the ground_truth table, which
    the live detection/alerts path never touches - kept isolated so the
    'labels hidden at inference' rule is enforced structurally, not just
    by convention."""
    total = db.query(func.count(GroundTruth.session_id)).scalar()
    counts = dict(
        db.query(GroundTruth.label, func.count(GroundTruth.session_id))
        .group_by(GroundTruth.label)
        .all()
    )
    normal = counts.get("normal", 0)
    anomaly_rate = round(100 * (1 - normal / total), 3) if total else 0.0
    return GroundTruthStatsOut(label_distribution=counts, anomaly_rate_pct=anomaly_rate)


def _alert_to_detail(alert: Alert, session: AccessSession) -> dict:
    return {
        "session_id": alert.session_id,
        "entity_id": alert.entity_id,
        "entity_type": session.entity_type,
        "risk_score": alert.risk_score,
        "predicted_type": alert.predicted_type,
        "reason": alert.reason,
        "created_at": alert.created_at,
        "timestamp": session.timestamp,
        "resource_accessed": session.resource_accessed,
        "auth_method": session.auth_method,
        "auth_result": session.auth_result,
        "source_ip": session.source_ip,
        "geo_city": session.geo_city,
        "geo_country": session.geo_country,
        "geo_lat": session.geo_lat,
        "geo_lon": session.geo_lon,
        "device_os": session.device_os,
        "device_fingerprint": session.device_fingerprint,
        "session_duration": session.session_duration,
    }


@app.get("/alerts", response_model=list[AlertDetailOut])
def list_alerts(
    predicted_type: Optional[str] = None,
    min_risk_score: float = 0.0,
    entity_id: Optional[str] = None,
    limit: int = Query(50, le=2000),
    offset: int = 0,
    db: OrmSession = Depends(get_db),
):
    """Ranked alert queue - real model output (populated by
    ml/populate_alerts.py), joined with session context so the dashboard
    gets everything it needs in one call."""
    q = db.query(Alert, AccessSession).join(
        AccessSession, Alert.session_id == AccessSession.session_id
    )
    if predicted_type:
        q = q.filter(Alert.predicted_type == predicted_type)
    if entity_id:
        q = q.filter(Alert.entity_id == entity_id)
    if min_risk_score:
        q = q.filter(Alert.risk_score >= min_risk_score)

    rows = q.order_by(Alert.risk_score.desc()).offset(offset).limit(limit).all()
    return [_alert_to_detail(alert, session) for alert, session in rows]


@app.get("/alerts/{session_id}", response_model=AlertDetailOut)
def get_alert(session_id: str, db: OrmSession = Depends(get_db)):
    row = (
        db.query(Alert, AccessSession)
        .join(AccessSession, Alert.session_id == AccessSession.session_id)
        .filter(Alert.session_id == session_id)
        .first()
    )
    if not row:
        return {"error": f"alert {session_id} not found"}
    alert, session = row
    return _alert_to_detail(alert, session)


@app.get("/devices/trust", response_model=list[DeviceTrustOut])
def device_trust(db: OrmSession = Depends(get_db)):
    """For each entity, compares its known device fingerprints (profile)
    against any device_spoofing-classified alerts against it, so the
    dashboard's Device Trust view can flag mismatches at a glance."""
    entities = db.query(Entity).all()
    spoof_alerts = db.query(Alert).filter(Alert.predicted_type == "device_spoofing").all()
    spoofed_by_entity = {}
    for a in spoof_alerts:
        session = db.query(AccessSession).filter(AccessSession.session_id == a.session_id).first()
        if session:
            spoofed_by_entity.setdefault(a.entity_id, set()).add(session.device_fingerprint)

    result = []
    for e in entities:
        known = e.device_fingerprints or []
        known_strs = [f"{os_}|{mac}|{proto}" for os_, mac, proto in known]
        flagged = list(spoofed_by_entity.get(e.entity_id, set()))
        result.append({
            "entity_id": e.entity_id,
            "entity_type": e.entity_type,
            "known_device_fingerprints": known_strs,
            "flagged_device_fingerprints": flagged,
            "trust_status": "flagged" if flagged else "trusted",
        })
    return result


@app.websocket("/ws/stream")
async def ws_stream(
    websocket: WebSocket,
    duration: float = Query(120.0, description="Demo duration in seconds to compress the full dataset into"),
    limit: int = Query(20000, description="Max sessions to replay"),
):
    await websocket.accept()
    try:
        await replay_sessions(websocket, demo_duration_seconds=duration, limit=limit)
        await websocket.close()
    except WebSocketDisconnect:
        pass
