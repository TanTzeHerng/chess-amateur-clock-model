# Clock Model Service

A small HTTP microservice that runs the **ChessMimic "clock model"** — a neural
network that predicts how long a human would think before making a chess move.
The main Chess Amateur app calls this service (in FIDE mode) to make the
engine's pacing feel human.

The service runs real PyTorch (CPU) inference on its own instance, so the
heavyweight model can live here without bloating the main app.

## What it does

Given a position (FEN), the recent moves, a target rating, and the current
clock state, the model produces a probability distribution over 30 "thinking
time" buckets. The service samples a bucket, then samples a concrete time from
that bucket's empirical distribution, and returns it.

The returned time is in the model's raw (blitz) scale. **This service does not
scale the value** — the main app is responsible for any up/down scaling to the
active time control.

## HTTP contract

### `POST /predict`

Request body (JSON):

```json
{
  "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 3",
  "recent_moves": ["e2e4", "e7e5", "g1f3", "b8c6"],
  "rating": 2300,
  "player_clock": 180.0,
  "opponent_clock": 175.0,
  "increment": 2.0
}
```

| field            | type            | notes                                        |
|------------------|-----------------|----------------------------------------------|
| `fen`            | string (req.)   | full 6-field FEN                             |
| `recent_moves`   | list of strings | UCI moves; last 12 are used; optional        |
| `rating`         | int             | player rating to simulate (default 1500)     |
| `player_clock`   | float seconds   | player's remaining time (default 300)        |
| `opponent_clock` | float seconds   | opponent's remaining time (default 300)      |
| `increment`      | float seconds   | per-move increment (default 0)               |

Response (`200`):

```json
{ "thinking_time": 11.0 }
```

Bad input (missing/empty `fen`, malformed `recent_moves`, non-numeric clocks,
illegal FEN, or an unknown UCI move) returns `400` with an `{"error": "..."}`
body. The worker never crashes on bad input.

### `GET /healthz`

```json
{ "status": "ok" }
```

## Running locally

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
pip install "flask>=3.0,<4.0" "gunicorn>=21.2,<24.0" "numpy>=1.24,<3.0"

# Place model artifacts under ./models/ (model.ckpt, scalers.pkl, clock_buckets.json)
PORT=8001 python server.py
# or with gunicorn:
PORT=8001 gunicorn --workers 1 --threads 4 --bind 0.0.0.0:8001 server:app
```

Artifact locations can be overridden with `CLOCK_MODEL_DIR`, `CLOCK_MODEL_PATH`,
`CLOCK_SCALERS_PATH`, and `CLOCK_BUCKETS_PATH`.

## Running the tests

```bash
python test_tokenizer.py
```

## Docker

The image installs CPU-only PyTorch and **downloads the model artifacts during
the build** (verifying the checkpoint is the real ~107 MB file and not a
Git-LFS pointer). Image tags must be lowercase.

```bash
docker build -t clock-model-service .
docker run --rm -p 8001:8001 clock-model-service
```

## How the main app uses it

The main Chess Amateur app reaches this service via a `CLOCK_MODEL_URL`
environment variable pointing at the service base URL, e.g.:

```
CLOCK_MODEL_URL=http://clock-model:8001
```

The app POSTs to `${CLOCK_MODEL_URL}/predict` and uses the returned
`thinking_time`, applying its own scaling for the active time control.

## Deployment notes

- **CPU-only.** The service forces CPU (`CHESSMIMIC_FORCE_CPU=true`).
- **Memory:** budget an instance with roughly **~1.5 GB RAM**. The checkpoint
  is ~107 MB on disk; PyTorch plus the loaded model and per-request tensors sit
  comfortably under that.
- **Concurrency:** the model is loaded once at startup and reused. A single
  gunicorn worker with a few threads is sufficient; scale out with more
  instances behind a load balancer if needed rather than many workers per
  instance (each worker loads its own model copy).
- **Startup:** the first request warms the model; `server.py` also warms it at
  boot when run directly.

## License & credit

This service uses the trained model and portions of the model/inference code
from the **ChessMimic** project by **Thomas Johnson**
(<https://github.com/thomasj02/1e4_ai>), which is licensed under the
**PolyForm Noncommercial License 1.0.0**. This deployment is a personal,
**noncommercial** hobby project, which that license permits.

Required Notice: Copyright 2026 Thomas Johnson (https://github.com/thomasj02/1e4_ai)

- The full license text is in [`LICENSE`](./LICENSE).
- Attribution and a file-by-file breakdown of derived vs. original code is in
  [`NOTICE`](./NOTICE).
- `tokenizer.py` is an original pure-Python reimplementation of the
  tokenization scheme (which itself originates in DeepMind's `searchless_chess`,
  Apache License 2.0); it does not copy the upstream C++ sources.

**Do not use this service or the bundled model for commercial purposes.**
