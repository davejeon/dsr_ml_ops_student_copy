#### The Dockerfile

```dockerfile
# Build from the repo root (see "build context" below):
#   docker build -f day2/Session_2/Dockerfile -t iris-svc:1.0 .

FROM python:3.11-slim            # ← base image (Debian slim, no extras)
WORKDIR /app                     # ← all subsequent commands run here

# ① Copy requirements FIRST — Docker caches each layer separately.
#   On later builds, if only service.py changed, the pip install layer
#   is served from cache. Reversing the order busts the cache every time.
COPY day2/Session_2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ② Copy only the artifacts the service actually needs
COPY day2/Session_1/service.py .
COPY day2/Session_1/iris_classifier.joblib .
COPY day2/Session_1/label_encoder.joblib .

EXPOSE 8000                      # ← documents the port; actual mapping is in `docker run`

# ③ Docker polls this during startup; the container turns "healthy" once it passes
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c \
      "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
```


#### .dockerignore — keep the context small

Sending a 2 GB `.venv/` directory to the daemon on every build is wasteful. A `.dockerignore` file (placed next to the build context root, i.e., the repo root) works like `.gitignore`:

```
# .dockerignore
.venv/
.git/
__pycache__/
*.py[cod]
*.ipynb
day1/Session_5/airflow_home/
*.db
```

This reduces the context from hundreds of MB to a few KB and cuts build time dramatically.

#### Multi-stage builds — shrinking the image

The build environment (pip, compilers, build-time headers) doesn't need to be in the final image. Use two stages:

```dockerfile
# ── Stage 1: install ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder
ENV VIRTUAL_ENV=/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY day2/Session_2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime (no pip, no build cache, no compilers) ──────────────────
FROM python:3.11-slim
ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV   # ← copy the venv, not pip itself

WORKDIR /app
COPY day2/Session_1/service.py .
COPY day2/Session_1/iris_classifier.joblib .
COPY day2/Session_1/label_encoder.joblib .

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c \
      "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### docker-compose for local development

`docker-compose.yml` encodes the `docker build` + `docker run` arguments so the whole setup is one command:

```yaml
services:
  iris-api:
    build:
      context: ../..
      dockerfile: day2/Session_2/Dockerfile
    image: iris-svc:1.0
    ports:
      - "8000:8000"
    environment:
      PYTHONUNBUFFERED: "1"
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 10s
      timeout: 3s
      retries: 3
      start_period: 5s
```


#### Anatomy of the GitHub workflow file

The file lives at `.github/workflows/ml-ci.yml` in the repo. GitHub scans that directory automatically — no registration or configuration needed.

```yaml
name: ml-ci          # display name shown in the GitHub UI

on:                  # TRIGGERS — when does this run?
  push:
    branches: ["main", "develop"]    # on any push to these branches
  pull_request:
    branches: ["main"]               # on any PR targeting main

jobs:
  ci:                          # job name (you can have multiple jobs)
    runs-on: ubuntu-latest     # the VM GitHub provides (free for public repos)

    steps:
      # Each step is one command. Steps run sequentially.
      # If any step exits non-zero, the job fails and later steps are skipped.

      - name: Check out code
        uses: actions/checkout@v4    # "uses" = run a pre-built action from the marketplace
                                     # This clones your repo onto the VM

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"               # cache the pip download cache between runs

      - name: Install dependencies
        run: pip install -r requirements.txt   # plain shell command

      - name: Train model
        run: python day2/Session_1/train.py    # CI trains the model itself

      - name: Run data tests
        run: pytest day2/Session_3/test_data.py -v

      - name: Run model tests
        run: pytest day2/Session_3/test_model.py -v
```

Key points:

| Concept | What it means |
|---------|--------------|
| `on:` | Declares triggers. You can also use `schedule:` for cron-based CT runs |
| `jobs:` | A workflow can have multiple jobs; by default they run in parallel |
| `needs:` | Makes a job wait for another to succeed (e.g. `needs: ci` on a deploy job) |
| `uses:` | Runs a reusable action from GitHub's marketplace (like a function call) |
| `run:` | Runs a raw shell command — anything you can type in a terminal |
| `with:` | Passes parameters to a `uses` action |
| `secrets.*` | Encrypted variables stored in GitHub settings — safe way to pass API keys |