"""
profiles.py
-----------
Builds per-entity "normal" behavioural profiles that the generator later
samples from to create benign traffic, and that anomaly injectors deviate
from to create labeled attack traffic.

Design rationale (documented for the report's "assumptions" section):
- Each entity (user / service_account / edge_device) gets a stable habitual
  profile: typical active hours, 1-2 home geo/IP locations, a resource pool
  it normally touches, a typical auth method, and 1-2 known device
  fingerprints.
- Service accounts behave differently from human users: tighter hour
  windows (often 24/7 or batch-window), narrower resource pools, and
  token/certificate auth instead of password.
- Edge devices behave differently again: fixed single location, single
  device fingerprint, machine-to-machine command sequences.
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple

from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

# ----------------------------------------------------------------------
# Reference catalogs
# ----------------------------------------------------------------------

RESOURCE_CATALOG = {
    "user": [
        "email", "sharepoint", "jira", "github", "confluence", "vpn",
        "hr_portal", "expense_system", "slack", "crm", "wiki",
        "finance_erp", "payroll", "code_repo", "ticketing_system",
    ],
    "service_account": [
        "database_prod", "database_staging", "message_queue", "s3_bucket",
        "api_gateway", "ci_cd_pipeline", "secrets_vault", "log_aggregator",
        "monitoring_stack", "backup_service",
    ],
    "edge_device": [
        "plc_controller", "scada_gateway", "sensor_hub", "camera_feed",
        "firmware_update_server", "telemetry_endpoint", "device_registry",
        "ot_historian",
    ],
}

SENSITIVE_RESOURCES = {
    "finance_erp", "payroll", "hr_portal", "secrets_vault",
    "database_prod", "scada_gateway", "plc_controller",
}

AUTH_METHODS = {
    "user": ["password", "biometric", "token"],
    "service_account": ["token", "certificate"],
    "edge_device": ["certificate", "token"],
}

OS_POOL = ["Windows 11", "Windows 10", "macOS 14", "Ubuntu 22.04",
           "RHEL 9", "iOS 17", "Android 14", "FirmwareOS 3.2"]

PROTOCOLS = ["HTTPS", "SSH", "RDP", "MQTT", "Modbus/TCP", "AMQP"]

# A pool of plausible "home" cities so impossible-travel and lateral
# movement geo maths stay realistic.
CITY_POOL = [
    ("Bengaluru", "India", 12.9716, 77.5946),
    ("Chennai", "India", 13.0827, 80.2707),
    ("Mumbai", "India", 19.0760, 72.8777),
    ("Singapore", "Singapore", 1.3521, 103.8198),
    ("London", "UK", 51.5074, -0.1278),
    ("Frankfurt", "Germany", 50.1109, 8.6821),
    ("New York", "USA", 40.7128, -74.0060),
    ("San Jose", "USA", 37.3382, -121.8863),
    ("Sydney", "Australia", -33.8688, 151.2093),
    ("Tokyo", "Japan", 35.6762, 139.6503),
    ("Dubai", "UAE", 25.2048, 55.2708),
    ("Sao Paulo", "Brazil", -23.5505, -46.6333),
]


@dataclass
class EntityProfile:
    entity_id: str
    entity_type: str            # user / service_account / edge_device
    home_locations: List[Tuple[str, str, float, float]]  # (city, country, lat, lon)
    active_hours: Tuple[int, int]   # (start_hour, end_hour), 24h clock
    resource_pool: List[str]
    typical_auth: str
    device_fingerprints: List[Tuple[str, str, str]]  # (os, mac, protocol)
    sessions_per_day: float
    avg_session_duration: float     # seconds
    role_sensitivity: str            # "standard" or "privileged" - drives which
                                      # resources are plausible for this entity


def _random_mac():
    return fake.mac_address()


def _make_device_fingerprint(entity_type: str) -> Tuple[str, str, str]:
    os_choice = random.choice(OS_POOL)
    if entity_type == "edge_device":
        os_choice = "FirmwareOS 3.2"
    protocol = random.choice(PROTOCOLS)
    return (os_choice, _random_mac(), protocol)


def build_entity_profile(entity_id: str, entity_type: str) -> EntityProfile:
    n_homes = 1 if entity_type != "user" else random.choices([1, 2], weights=[0.85, 0.15])[0]
    home_locations = random.sample(CITY_POOL, k=n_homes)

    if entity_type == "user":
        active_hours = random.choice([(8, 18), (7, 16), (9, 19), (6, 15)])
        sessions_per_day = random.uniform(3, 15)
        avg_duration = random.uniform(180, 1800)
        n_devices = random.choice([1, 1, 2])
    elif entity_type == "service_account":
        active_hours = (0, 24)  # effectively always-on / batch
        sessions_per_day = random.uniform(20, 200)
        avg_duration = random.uniform(5, 120)
        n_devices = 1
    else:  # edge_device
        active_hours = (0, 24)
        sessions_per_day = random.uniform(50, 500)
        avg_duration = random.uniform(1, 30)
        n_devices = 1

    resource_pool_full = RESOURCE_CATALOG[entity_type]
    k = min(len(resource_pool_full), random.randint(2, 5))
    resource_pool = random.sample(resource_pool_full, k=k)

    role_sensitivity = "privileged" if any(r in SENSITIVE_RESOURCES for r in resource_pool) \
        else random.choices(["standard", "privileged"], weights=[0.85, 0.15])[0]
    if role_sensitivity == "privileged" and entity_type == "user":
        # privileged users occasionally touch one sensitive resource normally
        extra = random.choice(list(SENSITIVE_RESOURCES - set(resource_pool)))
        resource_pool.append(extra)

    typical_auth = random.choice(AUTH_METHODS[entity_type])
    device_fingerprints = [_make_device_fingerprint(entity_type) for _ in range(n_devices)]

    return EntityProfile(
        entity_id=entity_id,
        entity_type=entity_type,
        home_locations=home_locations,
        active_hours=active_hours,
        resource_pool=resource_pool,
        typical_auth=typical_auth,
        device_fingerprints=device_fingerprints,
        sessions_per_day=sessions_per_day,
        avg_session_duration=avg_duration,
        role_sensitivity=role_sensitivity,
    )


def build_all_profiles(n_users=800, n_service_accounts=120, n_edge_devices=120):
    profiles = {}
    for i in range(n_users):
        eid = f"user_{i:04d}"
        profiles[eid] = build_entity_profile(eid, "user")
    for i in range(n_service_accounts):
        eid = f"svc_{i:04d}"
        profiles[eid] = build_entity_profile(eid, "service_account")
    for i in range(n_edge_devices):
        eid = f"dev_{i:04d}"
        profiles[eid] = build_entity_profile(eid, "edge_device")
    return profiles
