# syntax=docker/dockerfile:1.6
# =============================================================================
# LLM Hallucination Type Detector - production image
# Multi-stage build that keeps the final image small by:
#   1) building a wheel-compatible deps layer with a CPU-only PyTorch,
#   2) copying the runtime + model into a slim final image.
# =============================================================================

# ---------- 1. deps stage: install CPU-only torch + project deps -------------
FROM python:3.11-slim AS deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs needed by transformers/torch (none strictly required for
# CPU-only inference, but git is handy for some pip packages).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install CPU-only PyTorch first (saves ~1.5 GB vs. the default CUDA wheel),
# then the rest of the requirements without `torch` (override via env).
COPY requirements.txt /build/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch \
    && grep -vE '^\s*torch\s*$' /build/requirements.txt > /build/req_no_torch.txt \
    && pip install -r /build/req_no_torch.txt


# ---------- 2. runtime stage: app + model on a slim base ---------------------
FROM python:3.11-slim AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HALLUCINATION_MODEL_DIR=/app/models/saved_distilbert_hallucination_model

# Non-root user for safer runtime.
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app

# Copy installed Python packages from the deps stage.
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy the application source.
COPY --chown=app:app setup.py        /app/setup.py
COPY --chown=app:app run_api.py      /app/run_api.py
COPY --chown=app:app src             /app/src
COPY --chown=app:app requirements.txt /app/requirements.txt

# Copy the trained model checkpoint so the API boots with the real backend.
# (Uncomment the COPY line below or mount a volume to override at runtime.)
COPY --chown=app:app models/saved_distilbert_hallucination_model /app/models/saved_distilbert_hallucination_model

# Install the local `src/` package in editable mode so `src.api.main` resolves.
RUN pip install -e /app --no-deps

USER app

EXPOSE 8000

# Lightweight healthcheck using the /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# Run uvicorn bound to all interfaces so the container is reachable.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
