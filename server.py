"""Flask HTTP microservice for the ChessMimic clock model.

Exposes:
  POST /predict  -> {"thinking_time": float}
  GET  /healthz  -> {"status": "ok"}

The model is loaded once at process startup and reused across requests.
Runs on CPU. Read ``PORT`` from the environment (default 8001) and bind
0.0.0.0. Predicted thinking time is returned in the model's raw (blitz) scale;
the calling application is responsible for any up/down scaling.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, jsonify, request

from predictor import ClockPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clock_model_service")

# Model artifact locations. Overridable via env vars so the same code works
# in the Docker image (baked-in paths) and for local testing.
MODEL_DIR = Path(os.getenv("CLOCK_MODEL_DIR", str(Path(__file__).parent / "models")))
MODEL_PATH = Path(os.getenv("CLOCK_MODEL_PATH", str(MODEL_DIR / "model.ckpt")))
SCALERS_PATH = Path(os.getenv("CLOCK_SCALERS_PATH", str(MODEL_DIR / "scalers.pkl")))
BUCKETS_PATH = Path(os.getenv("CLOCK_BUCKETS_PATH", str(MODEL_DIR / "clock_buckets.json")))

app = Flask(__name__)

# Lazily-initialised singleton so importing this module (e.g. for a Docker
# build-time sanity import) does not require the artifacts to be present.
_predictor = None


def get_predictor() -> ClockPredictor:
    global _predictor
    if _predictor is None:
        logger.info("Loading clock model from %s", MODEL_PATH)
        _predictor = ClockPredictor(MODEL_PATH, SCALERS_PATH, BUCKETS_PATH)
        logger.info("Clock model loaded.")
    return _predictor


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    fen = data.get("fen")
    if not isinstance(fen, str) or not fen.strip():
        return jsonify({"error": "'fen' is required and must be a non-empty string"}), 400

    recent_moves = data.get("recent_moves", [])
    if recent_moves is None:
        recent_moves = []
    if not isinstance(recent_moves, list) or not all(isinstance(m, str) for m in recent_moves):
        return jsonify({"error": "'recent_moves' must be a list of UCI move strings"}), 400

    try:
        rating = int(data.get("rating", 1500))
        player_clock = float(data.get("player_clock", 300.0))
        opponent_clock = float(data.get("opponent_clock", 300.0))
        increment = float(data.get("increment", 0.0))
    except (TypeError, ValueError):
        return jsonify({"error": "rating/clock/increment must be numeric"}), 400

    if player_clock < 0 or opponent_clock < 0 or increment < 0:
        return jsonify({"error": "clock and increment values must be non-negative"}), 400

    try:
        thinking_time = get_predictor().predict_thinking_time(
            fen=fen,
            recent_moves=recent_moves,
            rating=rating,
            player_clock=player_clock,
            opponent_clock=opponent_clock,
            increment=increment,
        )
    except ValueError as exc:
        # Bad FEN, unknown move, etc. -> client error, worker stays alive.
        logger.warning("Bad prediction input: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 - never crash the worker
        logger.exception("Prediction failed")
        return jsonify({"error": "internal prediction error"}), 500

    return jsonify({"thinking_time": float(thinking_time)})


if __name__ == "__main__":
    # Warm the model so the first request is fast, then serve.
    port = int(os.getenv("PORT", "8001"))
    get_predictor()
    app.run(host="0.0.0.0", port=port)
