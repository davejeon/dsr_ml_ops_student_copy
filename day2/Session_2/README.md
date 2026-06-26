# Day 2 — Session 2: Containerizing the Iris Service

Packages the FastAPI inference service from Session 1 into a Docker image.

**Prerequisite:** run `train.py` in Session 1 to produce the `.joblib` files.

---

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Single-stage image recipe (good starting point) |
| `docker-compose.yml` | One-command build + run for local development |
| `requirements.txt` | Minimal runtime deps (no dev/notebook packages) |
| `containerize.ipynb` | Step-by-step walkthrough of the whole workflow |

The `.dockerignore` lives at the **repo root** (where the build context is set).

---

## Quick start

### Option A — docker run directly

```bash
# 1. Train the model (Session 1)
python day2/Session_1/train.py

# 2. Build the image (from repo root)
docker build -f day2/Session_2/Dockerfile -t iris-svc:1.0 .

# 3. Run the container
docker run --rm -p 8000:8000 --name iris-api iris-svc:1.0
```

### Option B — docker-compose (recommended)

```bash
# From day2/Session_2/
docker compose up --build          # build and start (blocks — shows logs)
docker compose up --build -d       # same but detached

docker compose down                # stop and remove
docker compose logs -f iris-api    # follow logs while running
```

---

## Test the running container

```bash
# Liveness
curl http://localhost:8000/health

# Readiness (503 until model loaded, 200 after)
curl http://localhost:8000/ready

# Inference
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"septal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Interactive API docs
open http://localhost:8000/docs
```

---

## Build context

Because `service.py` and the `.joblib` files live in `Session_1/` while the
`Dockerfile` is in `Session_2/`, the build context must be the **repo root** so
both directories are visible to Docker:

```
docker build -f day2/Session_2/Dockerfile -t iris-svc:1.0 .
                                                            ^
                                               repo root = build context
```

---