"""
schemas.py
----------
Pydantic response models for the API. Kept separate from the SQLAlchemy
models in database.py so the API contract (what clients see) is decoupled
from the storage schema (what's convenient to query).
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class SessionOut(BaseModel):
    session_id: str
    event_id: str
    entity_id: str
    entity_type: str
    timestamp: datetime
    source_ip: str
    geo_city: str
    geo_country: str
    geo_lat: float
    geo_lon: float
    resource_accessed: str
    auth_method: str
    auth_result: str
    session_duration: float
    command_sequence: str
    device_os: str
    device_mac: str
    protocol: str
    device_fingerprint: str

    class Config:
        from_attributes = True


class EntityOut(BaseModel):
    entity_id: str
    entity_type: str
    home_locations: list
    active_hours: list
    resource_pool: list
    typical_auth: str
    device_fingerprints: list
    sessions_per_day: float
    avg_session_duration: float
    role_sensitivity: str
    first_seen: Optional[datetime] = None
    session_count: int

    class Config:
        from_attributes = True


class StatsOut(BaseModel):
    total_sessions: int
    total_entities: int
    entity_type_breakdown: dict
    date_range: dict


class GroundTruthStatsOut(BaseModel):
    """Dev/eval-only view - never exposed to a 'live detection' client,
    only used by the evaluation module and internal dashboards."""
    label_distribution: dict
    anomaly_rate_pct: float


class AlertOut(BaseModel):
    session_id: str
    entity_id: str
    risk_score: float
    predicted_type: Optional[str] = None
    reason: Optional[dict] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AlertDetailOut(BaseModel):
    """Alert enriched with the underlying session's full context - what
    the dashboard's alert queue and alert-detail views actually need,
    rather than making the frontend stitch two API calls together."""
    session_id: str
    entity_id: str
    entity_type: str
    risk_score: float
    predicted_type: Optional[str] = None
    reason: Optional[dict] = None
    created_at: datetime
    timestamp: datetime
    resource_accessed: str
    auth_method: str
    auth_result: str
    source_ip: str
    geo_city: str
    geo_country: str
    geo_lat: float
    geo_lon: float
    device_os: str
    device_fingerprint: str
    session_duration: float


class DeviceTrustOut(BaseModel):
    entity_id: str
    entity_type: str
    known_device_fingerprints: list
    flagged_device_fingerprints: list
    trust_status: str  # "trusted" | "flagged"
