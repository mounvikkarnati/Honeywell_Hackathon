"""
anomalies.py
------------
Injects the anomaly / edge-case patterns from the "Behaviours to Simulate"
table into an entity's session history. Each function returns a list of
session dicts labeled with the specific anomaly_type (ground truth), which
is retained ONLY in the separate labels file - never exposed as a model
input feature.

Patterns implemented:
  - brute_force
  - impossible_travel
  - credential_stuffing
  - lateral_movement
  - device_spoofing
  - low_and_slow_exfiltration
  - insider_drift   (edge case, ambiguous - used for false-positive tuning)
"""

import math
import random
import uuid
from datetime import timedelta

import numpy as np
from faker import Faker

from .profiles import RESOURCE_CATALOG, CITY_POOL, _make_device_fingerprint
from .baseline import _command_sequence, _jitter_geo

fake = Faker()


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _base_event(profile, ts, resource, auth_method, auth_result, duration,
                 city, country, lat, lon, source_ip, os_, mac_, protocol,
                 anomaly_type, command_seq=None):
    return {
        "event_id": str(uuid.uuid4()),
        "entity_id": profile.entity_id,
        "entity_type": profile.entity_type,
        "timestamp": ts.isoformat(),
        "source_ip": source_ip,
        "geo_city": city,
        "geo_country": country,
        "geo_lat": round(lat, 4),
        "geo_lon": round(lon, 4),
        "resource_accessed": resource,
        "auth_method": auth_method,
        "auth_result": auth_result,
        "session_duration": round(duration, 1),
        "command_sequence": command_seq or _command_sequence(resource),
        "device_os": os_,
        "device_mac": mac_,
        "protocol": protocol,
        "device_fingerprint": f"{os_}|{mac_}|{protocol}",
        "label": anomaly_type,
    }


def inject_brute_force(profile, anchor_ts):
    """Rapid repeated failed-auth attempts from one source in a short window."""
    events = []
    city, country, lat, lon = random.choice(profile.home_locations)
    attacker_ip = fake.ipv4_public()
    n_attempts = random.randint(15, 60)
    window_seconds = random.randint(30, 300)
    os_, mac_, protocol = _make_device_fingerprint(profile.entity_type)
    for i in range(n_attempts):
        ts = anchor_ts + timedelta(seconds=(window_seconds / n_attempts) * i)
        success = (i == n_attempts - 1) and random.random() < 0.3  # rare eventual success
        events.append(_base_event(
            profile, ts, random.choice(profile.resource_pool), profile.typical_auth,
            "success" if success else "failure", random.uniform(1, 5),
            city, country, lat, lon, attacker_ip, os_, mac_, protocol,
            "brute_force",
        ))
    return events


def inject_impossible_travel(profile, anchor_ts):
    """Same entity_id logging in from geographically distant locations
    within an implausible time gap (implies travel speed >> commercial flight)."""
    home_city, home_country, home_lat, home_lon = random.choice(profile.home_locations)
    far_options = [c for c in CITY_POOL if _haversine_km(home_lat, home_lon, c[2], c[3]) > 3000]
    if not far_options:
        far_options = CITY_POOL
    city2, country2, lat2, lon2 = random.choice(far_options)

    gap_minutes = random.randint(5, 90)  # implausible for the distance involved
    ts1 = anchor_ts
    ts2 = anchor_ts + timedelta(minutes=gap_minutes)

    os_, mac_, protocol = random.choice(profile.device_fingerprints)
    os2, mac2, protocol2 = _make_device_fingerprint(profile.entity_type)

    e1 = _base_event(
        profile, ts1, random.choice(profile.resource_pool), profile.typical_auth,
        "success", random.uniform(60, 900), home_city, home_country, home_lat, home_lon,
        fake.ipv4_public(), os_, mac_, protocol, "impossible_travel",
    )
    e2 = _base_event(
        profile, ts2, random.choice(profile.resource_pool), profile.typical_auth,
        "success", random.uniform(60, 900), city2, country2, lat2, lon2,
        fake.ipv4_public(), os2, mac2, protocol2, "impossible_travel",
    )
    return [e1, e2]


def inject_credential_stuffing(all_profiles, anchor_ts, n_targets_range=(20, 80)):
    """Many entity_ids, few source_ips, high failure rate — same campaign,
    so this returns events across MULTIPLE entities sharing 1-3 source IPs."""
    n_targets = random.randint(*n_targets_range)
    victims = random.sample(list(all_profiles.values()), k=min(n_targets, len(all_profiles)))
    attacker_ips = [fake.ipv4_public() for _ in range(random.randint(1, 3))]
    events = []
    window_seconds = random.randint(300, 1800)
    for i, profile in enumerate(victims):
        ts = anchor_ts + timedelta(seconds=(window_seconds / len(victims)) * i)
        ip = random.choice(attacker_ips)
        success = random.random() < 0.05  # low hit rate, realistic for stuffing
        city, country, lat, lon = random.choice(CITY_POOL)  # attacker geo, not victim's home
        os_, mac_, protocol = _make_device_fingerprint(profile.entity_type)
        events.append(_base_event(
            profile, ts, random.choice(profile.resource_pool), profile.typical_auth,
            "success" if success else "failure", random.uniform(1, 5),
            city, country, lat, lon, ip, os_, mac_, protocol, "credential_stuffing",
        ))
    return events


def inject_lateral_movement(profile, anchor_ts, entity_type_for_resources="user"):
    """A compromised entity accessing an unusual sequence/breadth of
    resources it never touched before (outside its normal resource_pool)."""
    all_resources = [r for cat in RESOURCE_CATALOG.values() for r in cat]
    novel_resources = [r for r in all_resources if r not in profile.resource_pool]
    n_hops = random.randint(4, 9)
    chosen = random.sample(novel_resources, k=min(n_hops, len(novel_resources)))

    city, country, lat, lon = random.choice(profile.home_locations)
    os_, mac_, protocol = random.choice(profile.device_fingerprints)
    events = []
    for i, resource in enumerate(chosen):
        ts = anchor_ts + timedelta(minutes=i * random.uniform(2, 8))
        events.append(_base_event(
            profile, ts, resource, profile.typical_auth, "success",
            random.uniform(30, 300), city, country, lat, lon,
            fake.ipv4_public(), os_, mac_, protocol, "lateral_movement",
            command_seq="connect|enumerate|access_unfamiliar_resource",
        ))
    return events


def inject_device_spoofing(profile, anchor_ts):
    """A device_id reappearing with a mismatched fingerprint
    (different OS/MAC than history)."""
    known_os, known_mac, known_protocol = random.choice(profile.device_fingerprints)
    spoof_os = random.choice([o for o in
        ["Windows 11", "Windows 10", "macOS 14", "Ubuntu 22.04", "RHEL 9",
         "iOS 17", "Android 14", "FirmwareOS 3.2"] if o != known_os])
    spoof_mac = fake.mac_address()
    spoof_protocol = random.choice(["HTTPS", "SSH", "RDP", "MQTT", "Modbus/TCP", "AMQP"])

    city, country, lat, lon = random.choice(profile.home_locations)
    ts = anchor_ts
    return [_base_event(
        profile, ts, random.choice(profile.resource_pool), profile.typical_auth,
        "success", random.uniform(30, 300), city, country, lat, lon,
        fake.ipv4_public(), spoof_os, spoof_mac, spoof_protocol, "device_spoofing",
    )]


def inject_low_and_slow_exfiltration(profile, anchor_ts, n_days=14):
    """Gradual, small, off-hours resource access building up over days/weeks."""
    events = []
    all_resources = [r for cat in RESOURCE_CATALOG.values() for r in cat]
    sensitive_pool = [r for r in all_resources if r not in profile.resource_pool]
    city, country, lat, lon = random.choice(profile.home_locations)
    os_, mac_, protocol = random.choice(profile.device_fingerprints)

    for day in range(n_days):
        ts = anchor_ts + timedelta(days=day, hours=random.uniform(0, 4))  # off-hours (0-4am)
        resource = random.choice(sensitive_pool) if sensitive_pool else random.choice(profile.resource_pool)
        # volume/duration grows slightly over time to mimic gradual staging
        duration = random.uniform(20, 60) + day * random.uniform(2, 6)
        events.append(_base_event(
            profile, ts, resource, profile.typical_auth, "success", duration,
            city, country, lat, lon, fake.ipv4_public(), os_, mac_, protocol,
            "low_and_slow_exfiltration",
            command_seq="connect|list_files|read_file|compress|transfer_small_chunk",
        ))
    return events


def inject_insider_drift(profile, anchor_ts, n_days=20):
    """Legitimate entity slowly expanding privilege or resource footprint —
    ambiguous edge case, deliberately NOT a clean attack signature.
    Used for false-positive tuning, so keep this subtler/slower than
    low-and-slow exfiltration and give it plausible business cover
    (e.g. a role change) rather than attacker tradecraft."""
    events = []
    all_resources = [r for cat in RESOURCE_CATALOG.values() for r in cat]
    adjacent_pool = [r for r in all_resources if r not in profile.resource_pool]
    city, country, lat, lon = random.choice(profile.home_locations)
    os_, mac_, protocol = random.choice(profile.device_fingerprints)
    start_h, end_h = profile.active_hours if profile.active_hours != (0, 24) else (8, 18)

    n_new_resources = random.randint(1, 3)
    new_resources = random.sample(adjacent_pool, k=min(n_new_resources, len(adjacent_pool)))

    for day in range(n_days):
        if random.random() < 0.4:
            continue  # not every day - gradual, not constant
        ts = anchor_ts + timedelta(days=day, hours=random.uniform(start_h, end_h))
        # resource pool slowly widens over the window rather than jumping immediately
        if day < n_days * 0.3:
            resource = random.choice(profile.resource_pool)
        else:
            resource = random.choice(profile.resource_pool + new_resources)
        events.append(_base_event(
            profile, ts, resource, profile.typical_auth, "success",
            random.uniform(120, 900), city, country, lat, lon,
            fake.ipv4_public(), os_, mac_, protocol, "insider_drift",
        ))
    return events
