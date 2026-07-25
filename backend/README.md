# Backend (Phase 2)

FastAPI + SQLAlchemy (SQLite for the hackathon build; swap `DATABASE_URL`
for Postgres later with no code changes).

## Setup
```bash
pip install fastapi uvicorn sqlalchemy websockets

# 1. Generate data (Phase 1) if not already done
python3 -m generator.main --days 14 --users 300 --service-accounts 15 \
    --edge-devices 15 --anomaly-rate 0.02 --outdir ./output

# 2. Ingest into DB
python3 -m backend.ingest --outdir ./output --reset

# 3. Run the API
uvicorn backend.app:app --reload --port 8000
```

## Endpoints
| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness check |
| `GET /entities?entity_type=&limit=&offset=` | list entity profiles |
| `GET /entities/{entity_id}` | single entity profile |
| `GET /entities/{entity_id}/history` | session history for one entity |
| `GET /sessions?entity_id=&resource_accessed=&auth_result=` | filterable session log |
| `GET /stats` | dataset-level stats (entity-visible) |
| `GET /admin/ground-truth-stats` | dev/eval-only label distribution — **never** used by the detector, kept structurally isolated so labels stay hidden at inference |
| `GET /alerts` | **Phase 3 stub** — contract is stable now, detector will populate this table without changing the API shape |
| `WS /ws/stream?duration=&limit=` | simulated live event feed, replays stored sessions in timestamp order compressed into `duration` seconds |

## Design notes / known limitations
- SQLite chosen for zero-config demo speed; `DATABASE_URL` env var swaps to Postgres.
- Ground truth lives in its own table (`ground_truth`), joined only by
  `/admin` and eval code — never by `/alerts` or `/sessions` — so the
  "hidden at inference" rule is enforced by table isolation, not just convention.
- The `/ws/stream` compression is naive linear scaling of the full dataset
  span into `duration` seconds; at high session density this produces a
  near-continuous burst rather than a readable pace. Fine for now — worth
  tuning (e.g. minimum inter-event delay, or capping events/sec) once the
  dashboard's live-feed UI exists in Phase 4.
