"""
database.py
-----------
SQLAlchemy models + engine setup.

Using SQLite for the hackathon build (zero external server, zero config,
fast to demo). The schema is plain SQLAlchemy so swapping the connection
string to Postgres later is a one-line change (`DATABASE_URL`), not a
rewrite - documented here so it's an explicit, deliberate scope choice
rather than an oversight.

Tables:
  - entities        : one row per user / service_account / edge_device,
                       storing the profile summary (for the dashboard's
                       entity-history view and for cold-start lookups)
  - sessions        : one row per access event (the feature-visible data
                       the detection model consumes)
  - ground_truth    : label per session_id, kept in a SEPARATE table on
                       purpose - mirrors the "hidden at inference" schema
                       requirement. Only /admin and /stats endpoints (dev
                       & evaluation use) touch this table; /predict and
                       /alerts endpoints never join against it.
  - alerts          : populated later in Phase 3 once the detector exists.
                       Table is created now so the API contract is stable
                       across phases.
"""

import os
from datetime import datetime

from sqlalchemy import (Column, String, Float, Integer, DateTime, JSON,
                         create_engine, Index)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./cyberanomaly.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Entity(Base):
    __tablename__ = "entities"

    entity_id = Column(String, primary_key=True, index=True)
    entity_type = Column(String, index=True)
    home_locations = Column(JSON)
    active_hours = Column(JSON)
    resource_pool = Column(JSON)
    typical_auth = Column(String)
    device_fingerprints = Column(JSON)
    sessions_per_day = Column(Float)
    avg_session_duration = Column(Float)
    role_sensitivity = Column(String)
    first_seen = Column(DateTime, nullable=True)   # populated at ingest, used for cold-start logic
    session_count = Column(Integer, default=0)


class AccessSession(Base):
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True, index=True)
    event_id = Column(String, index=True)
    entity_id = Column(String, index=True)
    entity_type = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    source_ip = Column(String)
    geo_city = Column(String)
    geo_country = Column(String)
    geo_lat = Column(Float)
    geo_lon = Column(Float)
    resource_accessed = Column(String)
    auth_method = Column(String)
    auth_result = Column(String)
    session_duration = Column(Float)
    command_sequence = Column(String)
    device_os = Column(String)
    device_mac = Column(String)
    protocol = Column(String)
    device_fingerprint = Column(String)


Index("ix_sessions_entity_ts", AccessSession.entity_id, AccessSession.timestamp)


class GroundTruth(Base):
    __tablename__ = "ground_truth"
    session_id = Column(String, primary_key=True, index=True)
    event_id = Column(String, index=True)
    entity_id = Column(String, index=True)
    label = Column(String, index=True)   # normal / <anomaly_type>


class Alert(Base):
    __tablename__ = "alerts"
    session_id = Column(String, primary_key=True, index=True)
    entity_id = Column(String, index=True)
    risk_score = Column(Float, index=True)
    predicted_type = Column(String, nullable=True)
    reason = Column(JSON, nullable=True)          # explainability payload (Phase 5)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
