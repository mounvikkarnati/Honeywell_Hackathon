"""
baseline.py
-----------
Generates benign ("normal baseline") sessions for every entity across the
simulation window, sampled from that entity's habitual profile with noise.
This is the "Benign" signal type from the Behaviours-to-Simulate table.
"""

import random
import uuid
from datetime import datetime, timedelta

import numpy as np
from faker import Faker

fake = Faker()

COMMANDS_BY_RESOURCE = {
    "default": ["login", "view_dashboard", "read_file", "logout"],
    "code_repo": ["login", "git_pull", "git_push", "code_review", "logout"],
    "database_prod": ["connect", "query_select", "query_update", "disconnect"],
    "secrets_vault": ["connect", "read_secret", "disconnect"],
    "finance_erp": ["login", "view_ledger", "export_report", "logout"],
    "payroll": ["login", "view_payslip", "logout"],
    "scada_gateway": ["poll_status", "read_tag", "write_setpoint"],
    "plc_controller": ["read_registers", "write_registers"],
    "ci_cd_pipeline": ["trigger_build", "deploy", "rollback"],
}


def _command_sequence(resource: str) -> str:
    cmds = COMMANDS_BY_RESOURCE.get(resource, COMMANDS_BY_RESOURCE["default"])
    length = random.randint(2, min(4, len(cmds)))
    seq = [cmds[0]] + random.sample(cmds[1:], k=length - 1) if length > 1 else [cmds[0]]
    return "|".join(seq)


def _jitter_geo(lat, lon, max_km=15):
    # ~0.009 degrees latitude per km
    dlat = random.uniform(-max_km, max_km) * 0.009
    dlon = random.uniform(-max_km, max_km) * 0.009
    return lat + dlat, lon + dlon


def generate_baseline_sessions(profile, sim_start: datetime, sim_days: int):
    """Yield benign session dicts for one entity across the sim window."""
    sessions = []
    for day_offset in range(sim_days):
        day = sim_start + timedelta(days=day_offset)
        n_sessions = np.random.poisson(profile.sessions_per_day)
        for _ in range(n_sessions):
            start_h, end_h = profile.active_hours
            if start_h == 0 and end_h == 24:
                hour = random.randint(0, 23)
            else:
                # mostly within active hours, small chance just outside (realistic noise)
                if random.random() < 0.94:
                    hour = random.randint(start_h, max(start_h, end_h - 1))
                else:
                    hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            ts = day.replace(hour=hour, minute=minute, second=second)

            city, country, lat, lon = random.choice(profile.home_locations)
            lat_j, lon_j = _jitter_geo(lat, lon)
            source_ip = fake.ipv4_public()

            resource = random.choice(profile.resource_pool)
            auth_method = profile.typical_auth
            os_, mac_, protocol = random.choice(profile.device_fingerprints)
            duration = max(1, np.random.lognormal(
                mean=np.log(max(profile.avg_session_duration, 2)), sigma=0.4))

            sessions.append({
                "event_id": str(uuid.uuid4()),
                "entity_id": profile.entity_id,
                "entity_type": profile.entity_type,
                "timestamp": ts.isoformat(),
                "source_ip": source_ip,
                "geo_city": city,
                "geo_country": country,
                "geo_lat": round(lat_j, 4),
                "geo_lon": round(lon_j, 4),
                "resource_accessed": resource,
                "auth_method": auth_method,
                "auth_result": "success",
                "session_duration": round(duration, 1),
                "command_sequence": _command_sequence(resource),
                "device_os": os_,
                "device_mac": mac_,
                "protocol": protocol,
                "device_fingerprint": f"{os_}|{mac_}|{protocol}",
                "label": "normal",
            })
    return sessions
