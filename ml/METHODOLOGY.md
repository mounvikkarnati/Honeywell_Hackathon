# Phase 3 — ML Models & Evaluation: Methodology & Results

## 1. Baseline profiling model (Deliverable #2)
Implemented as a **rolling-window, per-entity statistical profile**, learned
directly from observed access logs — never from the generator's internal
"ground truth" profile object. Three requirements are satisfied by one
design choice:

- **No leakage**: each session's features only ever reference that
  entity's *prior* events (`ml/profiler.py`, chronological single-pass
  scan).
- **Concept drift**: the rolling window is 5 days by default (configurable),
  so behaviour older than that "ages out" automatically — a new work
  pattern stops being flagged once it's been the norm for a few days.
- **Cold-start**: entities with fewer than `MIN_HISTORY_FOR_OWN` (5) prior
  sessions in their window fall back to peer-group stats (same
  `entity_type`), rather than computing unstable statistics from 1-2 data
  points. Verified: 1.5% of all sessions (1,650/108,989) hit this
  fallback and still produce sensible, non-degenerate feature values.

A second pass (`ml/ip_features.py`) computes the same kind of rolling
stats grouped by `source_ip` instead of by entity — this is what actually
catches brute force (many failures, one entity/IP) and credential
stuffing (many entities, one/few IPs), patterns a purely per-entity view
can't see on its own.

## 2. Feature set (15 engineered signals, `ml/features.py`)
Cold-start flag, log(prior session count), new-resource flag, new-device
flag, hour-of-day z-score, off-hours flag, log(time gap since last
session), log(implied travel speed), session-duration z-score, auth
failure flag, rolling failure rate, log(IP-recent failure count),
log(IP-recent event count), log(IP-recent distinct entities), and a
resource-sensitivity flag (domain knowledge of which resources are
sensitive — known upfront, not label leakage).

## 3. Detection model (Deliverable #3): Isolation Forest
Chosen as the Tier-1 required baseline — genuinely unsupervised (doesn't
need labeled anomalies to train, matching the real constraint that true
intrusions are a tiny, mostly-unknown-shape fraction of events).
`contamination=0.02`, 300 estimators.

## 4. Attack-type classifier (Deliverable #4): Random Forest
Trained only on non-"normal" rows (typing an attack is meaningless for
benign traffic), `class_weight="balanced"` since the attack mix is itself
imbalanced by design (brute force is common, device spoofing is rare).

## 5. Evaluation methodology
**Chronological 70/30 train/test split** (not random) — matches how this
would actually be deployed: train on history, evaluate on what comes
next. Train anomaly rate 1.93%, test anomaly rate 2.18% (close, confirms
the split didn't accidentally skew the label distribution).

All metrics computed via the standalone `evaluation/` module — nothing
hand-calculated inline, so numbers can never silently drift from what the
report/dashboard show.

## 6. Results (test split, 32,697 rows, 714 true anomalies)

### Binary detection
| Metric | Value |
|---|---|
| Precision | 0.984 |
| Recall | 0.700 |
| F1 | 0.818 |
| ROC-AUC | 0.979 |
| PR-AUC | 0.863 |
| False positive rate | 0.00025 (8 FP / 31,983 true negatives) |
| Precision @ top 1% alert budget | 1.00 (captures 46% of all anomalies) |
| Precision @ top 2% alert budget | 0.81 (captures 74% of all anomalies) |
| Precision @ top 5% alert budget | 0.40 (captures 92% of all anomalies) |

### Attack-type classification (oracle — evaluated on true anomalies only)
| Attack type | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| brute_force | 0.99 | 1.00 | 0.99 | 249 |
| credential_stuffing | 1.00 | 0.99 | 0.99 | 243 |
| device_spoofing | 1.00 | 1.00 | 1.00 | 7 |
| lateral_movement | 0.96 | 0.90 | 0.93 | 60 |
| low_and_slow_exfiltration | 0.80 | 0.69 | 0.74 | 89 |
| impossible_travel | 0.67 | 0.60 | 0.63 | 10 |
| insider_drift | 0.57 | 0.75 | 0.65 | 56 |
| **Overall accuracy** | | | **0.923** | 714 |
| **Weighted avg** | 0.93 | 0.92 | 0.92 | 714 |

**Why the spread is meaningful, not noise**: the three near-perfect
categories (brute force, credential stuffing, device spoofing) have loud,
structurally distinctive signatures (failure bursts, cross-entity IP
reuse, fingerprint mismatch). `insider_drift` scoring lowest (F1 0.65) is
*intentional validation* of the synthetic data design — it was built
specifically as an ambiguous edge case for false-positive tuning, not a
"win" case, per its description in `generator/ASSUMPTIONS.md`.
`impossible_travel`'s weaker score is largely a small-sample effect (only
10 test examples).

### Top classifier feature importances
1. `log_implied_speed_kmh` (0.166) — the impossible-travel signal
2. `new_device_flag` (0.133) — spoofing / lateral movement
3. `log_time_gap_seconds` (0.120) — brute-force rapid-fire
4. `new_resource_flag` (0.095) — lateral movement
5. `duration_zscore` (0.082)
6. `hour_zscore` (0.071) — drift / off-hours exfiltration

This ranking matches domain intuition exactly, which is a useful sanity
check that the model learned real signal rather than a spurious
correlation.

## 7. Known limitations (for the report)
- Risk-score scaling (0-100) is min-max normalized per evaluation batch;
  a live streaming deployment should pin this to a fixed reference range
  computed at training time, not recompute min/max per scoring call.
- The attack-type classifier is evaluated in an "oracle" setting (given
  the true anomalies) to isolate classification quality from detection
  recall. An end-to-end number (classifier accuracy on the detector's
  *actual* flagged set, including its false positives/negatives) is a
  natural next check before the final demo.
- `impossible_travel` and `device_spoofing` have small test-set sample
  sizes (10 and 7 respectively) — their per-class metrics should be read
  with that in mind, not treated as tightly estimated.
