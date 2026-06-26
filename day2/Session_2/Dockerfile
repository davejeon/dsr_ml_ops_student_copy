# =============================================================================
# Dockerfile — iris classifier inference service
# =============================================================================
#
# Build context MUST be the repo root so Docker can see both Session_1 and
# Session_2 directories.
#
# Quick start:
#   # From repo root
#   docker build -f day2/Session_2/Dockerfile -t iris-svc:1.0 .
#   docker run --rm -p 8000:8000 --name iris-api iris-svc:1.0
#
#   # Or from day2/Session_2/
#   docker build -f Dockerfile -t iris-svc:1.0 ../..
#
# Prerequisites: run day2/Session_1/train.py first to produce the .joblib files.
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# ── Layer 1: dependencies (cached separately from code) ──────────────────────
# Copy requirements.txt before any source code. If only service.py changes,
# Docker skips this layer on the next build and reuses the cached pip layer.
COPY day2/Session_2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 2: model artifacts and service code ─────────────────────────────────
# Copy only what the service needs at runtime — not the full repo.
COPY day2/Session_1/service.py .
COPY day2/Session_1/iris_classifier.joblib .
COPY day2/Session_1/label_encoder.joblib .

# ── Runtime config ────────────────────────────────────────────────────────────
EXPOSE 8000

# Docker polls /health during startup; the container enters "healthy" state
# once this command exits 0.  Kubernetes liveness probes work the same way.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c \
      "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
