# Cyber Anomaly Synthetic Data Generator

Deliverable 1 of 7 for the AI-Powered Behavioral Anomaly Detection challenge.

## Quick start
```bash
pip install faker numpy pandas
python3 -m generator.main --days 14 --users 300 --service-accounts 15 \
    --edge-devices 15 --anomaly-rate 0.02 --outdir ./output
```

All parameters are tunable. Scale up (`--users 800 --days 30`) for a final
run once the rest of the pipeline is ready to consume more data — just
watch memory (1.5M+ sessions caused an OOM kill in this environment).

## Files produced in `output/`
- `access_logs.csv` / `access_logs.json` — what the detection model sees (NO label column)
- `access_logs_full.csv` — same data WITH ground-truth label, for training/eval only
- `labels.csv` — event_id -> label mapping, kept separate per spec
- `entity_profiles.json` — serialized per-entity behavioural profiles (for dashboard entity-history view)
- `injection_episode_log.json` — every injected anomaly episode with pattern + timestamp
- `dataset_summary.json` — label distribution / anomaly rate sanity check
- `access_logs_sample.json` — first 200 rows, human-readable, for quick inspection

See `generator/ASSUMPTIONS.md` for the full documented behavioural
assumptions and attack taxonomy (required by Deliverable 1).

## Module layout
- `generator/profiles.py` — builds per-entity habitual behaviour profiles
- `generator/baseline.py` — generates benign traffic from those profiles
- `generator/anomalies.py` — 7 injector functions, one per attack pattern
- `generator/main.py` — orchestrates generation + writes all outputs
