# Synthetic Data Generator — Behavioural Assumptions & Attack Taxonomy

## 1. Entity model
Three entity types, each with a distinct habitual profile so "normal" is
genuinely different per entity type (not a single global baseline):

| entity_type | active hours | sessions/day | typical auth | resource pool size |
|---|---|---|---|---|
| user | fixed window, e.g. 8-18 | 3–15 | password / biometric / token | 2–5 (+1 sensitive if privileged) |
| service_account | 24/7 (batch/always-on) | 20–200 | token / certificate | 2–5 |
| edge_device | 24/7 | 50–500 | certificate / token | 2–5 |

Each entity additionally has: 1–2 home geo-locations, 1–2 known device
fingerprints (OS, MAC, protocol), and a habitual resource pool sampled
from a domain-relevant catalog (IT for users/service accounts, OT for
edge devices). ~15% of users are marked "privileged" and habitually touch
one sensitive resource (finance_erp, payroll, secrets_vault, etc.) as part
of their *normal* baseline — this is what makes lateral movement/insider
drift genuinely hard to separate from legitimate access.

## 2. Baseline ("Benign") generation
Sessions are sampled per entity per day (Poisson-distributed count around
that entity's `sessions_per_day`), with timestamp, geo (jittered ±15km
around a home location), resource, auth method, and device fingerprint
all drawn from that entity's habitual profile plus small noise (~6% chance
of an off-hours session, occasional secondary home location). This noise
is intentional: a zero-noise baseline makes every anomaly trivially
separable, which would not reflect a realistic detection problem.

## 3. Attack taxonomy (7 patterns injected)

| Pattern | How it's simulated | Ground-truth label |
|---|---|---|
| **Brute force** | 15–60 failed-auth attempts from one attacker IP against one entity within a 30–300s window; rare eventual success (~30% chance on the last attempt) | `brute_force` |
| **Impossible travel** | Same entity logs in successfully from its home location, then again 5–90 minutes later from a location >3,000km away — a travel speed no commercial flight can achieve | `impossible_travel` |
| **Credential stuffing** | 20–80 *different* entities targeted from a shared pool of 1–3 attacker IPs within a single campaign window; ~5% success rate (low hit-rate, high failure volume) — this is the one pattern that spans multiple entities, not one | `credential_stuffing` |
| **Lateral movement** | A single entity accesses 4–9 resources entirely outside its historical resource pool in rapid succession (minutes apart), simulating post-compromise enumeration | `lateral_movement` |
| **Device spoofing** | A known entity_id reappears but with an OS/MAC/protocol combination that doesn't match any of its known device fingerprints | `device_spoofing` |
| **Low-and-slow exfiltration** | One access per day (off-hours, 12am–4am) to a not-previously-used resource over ~14 days, with session duration/volume gradually increasing — deliberately slower and quieter than lateral movement | `low_and_slow_exfiltration` |
| **Insider drift** (edge case) | A legitimate entity gradually (not every day, ~60% of days) starts touching 1–3 adjacent resources during its *normal* active hours, with the widened footprint only appearing in the back half of the observation window. No attacker IP, no odd device, no odd hours — this is intentionally the hardest pattern to separate from a genuine role change, and is meant for false-positive-rate tuning rather than as a "win" case for the detector | `insider_drift` |

## 4. Injection rate & class imbalance
Anomalies are injected episode-by-episode (not per-event) until the total
anomalous *event* count reaches the configured target rate (default 2%,
tunable 0.5–3% per the spec) relative to total sessions. Pattern mix is
weighted (brute force and impossible travel are common but short episodes;
insider drift and low-and-slow are rarer but longer-running) to keep the
label distribution realistic and imbalanced, per the spec's explicit
"extreme class imbalance" requirement — this is deliberate, not an
oversight to fix later.

## 5. What's hidden at inference
`label` (ground truth) is stripped from `access_logs.csv` /
`access_logs.json` — the files the detection model actually consumes.
It is retained only in `labels.csv`, keyed by `event_id`/`session_id`, for
training and offline evaluation, matching the schema note: *"label —
normal / anomaly_type (for training and evaluation only — hidden at
inference)."*

## 6. Known limitations (for the report)
- Geo-coordinates are jittered around 12 fixed reference cities rather
  than drawn from a full geocoding service — sufficient for
  geo-velocity/impossible-travel math, not for fine-grained IP geolocation.
- `command_sequence` is a small templated action vocabulary per resource
  type, not a full session-replay log — adequate for sequence-model
  input, not a substitute for real command telemetry.
- Credential stuffing is the only cross-entity pattern; all other
  anomalies are single-entity episodes for simplicity of ground-truth
  bookkeeping.
- Default scale (300 users / 15 service accounts / 15 edge devices / 14
  days ≈ 109k sessions) is sized for fast iteration during a time-boxed
  build. All counts are CLI-configurable (`--users`, `--service-accounts`,
  `--edge-devices`, `--days`, `--anomaly-rate`) to scale up for a final run.
