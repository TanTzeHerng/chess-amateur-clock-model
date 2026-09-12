# Clock-model microservice (ChessMimic clock model served over HTTP).
# CPU-only PyTorch inference. Model artifacts are downloaded at build time so
# the running image is self-contained.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHESSMIMIC_FORCE_CPU=true \
    PORT=8001 \
    CLOCK_MODEL_DIR=/app/models

WORKDIR /app

# curl is needed to download the model artifacts during the build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (from the dedicated CPU index) so we do not
# drag in the large CUDA wheels, then the rest of the requirements.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
COPY requirements.txt .
# torch is already installed above; install the remaining deps (torch line is
# satisfied by the pre-installed CPU wheel).
RUN pip install "flask>=3.0,<4.0" "gunicorn>=21.2,<24.0" "numpy>=1.24,<3.0"

# ---------------------------------------------------------------------------
# Download model artifacts (PolyForm Noncommercial License 1.0.0 - see NOTICE).
# The checkpoint is stored in Git-LFS; fetch the real bytes from the LFS media
# host and FAIL THE BUILD if we accidentally got the ~134-byte LFS pointer.
# ---------------------------------------------------------------------------
ARG MODEL_BASE=https://media.githubusercontent.com/media/thomasj02/1e4_ai/master/backend/models/clock_model/2200_3500_brier
ARG RAW_BASE=https://raw.githubusercontent.com/thomasj02/1e4_ai/master/backend/models/clock_model/2200_3500_brier
ARG EXPECTED_CKPT_BYTES=107535521

RUN mkdir -p /app/models \
    && curl -fsSL -o /app/models/model.ckpt "${MODEL_BASE}/model.ckpt" \
    && curl -fsSL -o /app/models/scalers.pkl "${RAW_BASE}/scalers.pkl" \
    && curl -fsSL -o /app/models/clock_buckets.json "${RAW_BASE}/clock_buckets.json" \
    && ACTUAL=$(stat -c%s /app/models/model.ckpt) \
    && echo "model.ckpt size: ${ACTUAL} bytes (expected ~${EXPECTED_CKPT_BYTES})" \
    && if [ "${ACTUAL}" -lt 100000000 ]; then \
         echo "ERROR: model.ckpt is only ${ACTUAL} bytes - looks like a Git-LFS pointer, not the model." >&2; \
         exit 1; \
       fi

# Application code.
COPY tokenizer.py model.py clock_bucket_utils.py predictor.py server.py ./
COPY LICENSE NOTICE README.md ./

# Build-time sanity check: torch imports, the tokenizer works, and the app
# module imports cleanly. Fails the build early if anything is broken.
RUN python -c "import torch; import numpy; import tokenizer; import model; import clock_bucket_utils; import predictor; import server; \
t = tokenizer.tokenize('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'); \
assert len(t) == 78 and t[0] == 29, t; \
assert tokenizer.NUM_ACTIONS == 1968; \
print('build sanity OK: torch', torch.__version__, 'tokens', len(t))"

EXPOSE 8001

# Single gunicorn worker: the model is loaded once at startup and reused.
# Shell form so ${PORT} is expanded at runtime.
CMD gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT} server:app
