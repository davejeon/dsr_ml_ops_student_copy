# MLOps Learning Manual — Preparation Guide

A self-study reference synthesized from the 2-day MLOps course materials. Read this to build a solid conceptual foundation before (or alongside) the hands-on labs.

---

## Table of Contents

1. [What is MLOps?](#1-what-is-mlops)
2. [The ML Lifecycle](#2-the-ml-lifecycle)
3. [Reproducibility](#3-reproducibility)
4. [Experiment Tracking](#4-experiment-tracking)
5. [Data Versioning & Validation](#5-data-versioning--validation)
6. [Packaging & Serving Models](#6-packaging--serving-models)
7. [CI/CD for Machine Learning](#7-cicd-for-machine-learning)
8. [Monitoring & Drift Detection](#8-monitoring--drift-detection)
9. [Governance, Security & Team Structure](#9-governance-security--team-structure)
10. [The Non-Negotiables (Cheat Sheet)](#10-the-non-negotiables)
11. [Key Tools Reference](#11-key-tools-reference)
12. [Further Reading](#12-further-reading)

---

## 1. What is MLOps?

**MLOps** is the set of practices, tools, and culture that aim to deploy and maintain machine learning models in production reliably and efficiently.

It combines three disciplines:

- **Machine Learning / Data Science** — building models
- **DevOps / Software Engineering** — reliable software delivery (see below)
- **Data Engineering** — managing data at scale

### What is DevOps?

**DevOps** is a set of practices that combines software **Dev**elopment and IT **Op**erations. Its goal is to shorten the development lifecycle and deliver software reliably and continuously. Core ideas include:

- **Continuous Integration (CI)** — developers merge code frequently; each merge triggers automated builds and tests.
- **Continuous Delivery/Deployment (CD)** — code that passes CI is automatically packaged and deployed to production (or made ready for one-click deploy).
- **Infrastructure as Code** — servers and environments are defined in version-controlled config files (Terraform, Ansible), not set up manually.
- **Monitoring & feedback loops** — production systems are instrumented so teams learn quickly when something breaks.
- **Collaboration** — developers and operations engineers share ownership of the full lifecycle rather than throwing code "over the wall."

In short: DevOps ensures software goes from a developer's laptop to users quickly, safely, and repeatably. MLOps extends these ideas to the unique challenges of machine learning (data, models, drift).

### Why it matters

The ML model code is a **tiny fraction** of a real ML system (Sculley et al., Google 2015). Without MLOps:

- Models can't be reproduced ("works on my laptop")
- Models silently degrade as data changes (drift)
- No one knows which model version is in production
- Retraining is manual and error-prone
- Compliance and audit are impossible

### MLOps vs DevOps

### What is an "artifact"?

In software and ML, an **artifact** is any tangible output produced during the development process that you need to keep, version, or deploy. Examples:

- A compiled binary or Docker image (software artifact)
- A trained model file like `model.pkl` or `model.onnx` (ML artifact)
- A dataset snapshot (data artifact)
- A plot, report, or evaluation result (analysis artifact)

The **primary artifact** is the main deliverable of your workflow — in traditional software that's the application code/binary; in ML it's the code *plus* the trained model *plus* the data that produced it.

| Aspect | DevOps | MLOps |
|--------|--------|-------|
| Primary artifact | Code | Code + data + model |
| Tests | Unit, integration | + data validation, model quality |
| Versioning | Source code | + data + model + experiment |
| CI/CD | Build → Test → Deploy | + train → evaluate → deploy → monitor |
| Failure modes | Bugs, outages | + drift, bias, silent quality decay |
| Determinism | Mostly deterministic | Stochastic (seeds, hardware, data) |

### Maturity Levels (Google's framework)

| Level | Description | When appropriate |
|-------|-------------|-----------------|
| **Level 0** | Manual: scripts, notebooks, manual deploys | Proof of concept, early exploration |
| **Level 1** | ML pipeline automation: automated training, reproducible runs | Models with regular retraining needs |
| **Level 2** | CI/CD pipeline automation: code/data/model changes trigger automated retrain + deploy | High-velocity teams, many models |

**Key insight:** You don't always need Level 2. Pick the lowest level that solves your business pain.

---

## 2. The ML Lifecycle

```
   ┌──────────┐   ┌──────────┐   ┌─────────────┐   ┌──────────┐   ┌──────────┐
   │   Data   │──▶│ Features │──▶│   Train /   │──▶│  Deploy  │──▶│ Monitor  │
   │ (collect,│   │          │   │  Evaluate   │   │  (serve) │   │ (drift,  │
   │  label)  │   │          │   │             │   │          │   │  perf)   │
   └──────────┘   └──────────┘   └─────────────┘   └──────────┘   └────┬─────┘
        ▲                                                              │
        └──────────────────────── retrain / iterate ───────────────────┘
```

Each stage has its own failure modes:

| Stage | Common failure |
|-------|---------------|
| Data | Schema changes, label definition drift, missing values |
| Features | Train/serve skew, stale features |
| Training | Non-reproducible results, overfitting |
| Deployment | Wrong model version, configuration mismatch |
| Monitoring | Silent degradation, alert fatigue |

### What is feature engineering?

**Feature engineering** is the process of transforming raw data into input variables ("features") that a model can learn from effectively. Examples:

- Converting a date-of-birth column into an `age` integer.
- Encoding a categorical column ("red", "blue", "green") into numeric dummy variables.
- Computing `days_since_last_purchase` from a timestamp.
- Normalizing income to a 0–1 range so it doesn't dominate other features.

Good features often matter more than choosing a fancier algorithm. In production, feature engineering code must run identically during training and serving (see "train/serve skew" in Section 5).

### Data drift and model degradation

**Data drift** occurs when the statistical properties of the input data change over time compared to what the model was trained on. For example, if your model learned from users aged 25–45 but a marketing campaign suddenly brings in teenagers, the input distribution $P(X)$ has shifted.

**Model degradation** (also called model decay) is the consequence: as the real world drifts away from training conditions, the model's predictions become less accurate — even though nothing in the model's code has changed. The model isn't "broken"; the world simply no longer matches its assumptions.

Degradation can be:
- **Sudden** — a data pipeline breaks or a business rule changes overnight.
- **Gradual** — user behaviour slowly shifts over weeks/months.
- **Recurring** — seasonal patterns (holiday shopping, tax season).

The fix is monitoring (Section 8) combined with automated or scheduled retraining.

---

## 3. Reproducibility

### The Four Pillars

| Pillar | What to control | Tools |
|--------|----------------|-------|
| **Code** | Source, version, review | `git`, pull requests |
| **Environment** | Python version, library versions, OS deps | `venv`, `pip-tools`, `uv`, Docker |
| **Data** | Dataset version, schema, splits | DVC, LakeFS, S3 versioning |
| **Randomness** | Seeds, algorithm determinism | Explicit seed-setting, documentation |

### Environment management best practices

1. **Never rely on system Python** — always use a virtual environment
2. **Pin versions strictly** — `pip freeze` is a start; `pip-tools` or `uv` gives you locked dependency resolution
3. **Containerize** for cross-environment consistency (see Section 6)

### Seed setting pattern

```python
import os, random, numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
# For PyTorch:
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
```

Full determinism is hard (GPU non-determinism, parallel data loaders). Document what you can guarantee.

### Project structure

```
project/
├── data/            # raw, interim, processed (gitignored)
├── notebooks/       # exploration only
├── src/             # importable Python package
│   ├── data/
│   ├── features/          # feature engineering code (transforms, pipelines, encoders)
│   ├── models/
│   └── inference/
├── tests/
├── pyproject.toml
└── README.md
```

**Rule:** Notebooks for exploration; production code lives in `src/` and is unit-tested.

---

## 4. Experiment Tracking

### The problem it solves

Without tracking you get: `model_final.pkl`, `model_final_v2.pkl`, `model_final_v2_REAL.pkl` — and no one can answer "what hyperparameters gave us 0.91 AUC last Tuesday?"

### What an experiment tracker stores per run

- **Parameters** — hyperparameters, dataset version, code commit
- **Metrics** — loss, accuracy, AUC, latency
- **Artifacts** — model file, plots, confusion matrix
- **Tags / metadata** — author, git SHA, environment

### MLflow — the core components

| Component | Purpose |
|-----------|---------|
| **Tracking** | Log params, metrics, artifacts per run |
| **Projects** | Package code in a reproducible form |
| **Models** | Standard format for packaging models |
| **Model Registry** | Version management with stages (Staging → Production) |

### Minimal MLflow usage

```python
import mlflow

mlflow.set_experiment("churn-baseline")
with mlflow.start_run():
    mlflow.log_param("C", 1.0)
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_metric("auc", 0.87)
    mlflow.log_metric("accuracy", 0.83)
    mlflow.sklearn.log_model(model, "model")
```

### The tracking-to-production workflow

```
log runs ──▶ compare runs ──▶ pick best ──▶ register model ──▶ Staging ──▶ Production
```

---

## 5. Data Versioning & Validation

### Why version data?

Models are functions of **data + code + config**. If data changes silently, results are not reproducible.

Examples of silent data change:
- Upstream team renames a column
- A new country is added to the source
- Label definition changes ("active user" now means 30 days, not 7)

### Approaches

| Approach | Tools | Trade-offs |
|----------|-------|------------|
| Snapshots in object storage | S3 versioning | Simple, no metadata |
| Git-like data versioning | DVC, LakeFS | Diffs, branches, ties to git |
| Data warehouse time travel | Snowflake, BigQuery, Delta Lake | Powerful, vendor-bound |
| Feature stores | Feast, Tecton | Reuse + consistency train/serve |

### Data validation

Before training **and** before serving, validate:

- **Schema** — columns present, correct types
- **Distribution** — means, ranges, null rates within expected bounds
- **Referential integrity** — foreign keys valid

Tools: Great Expectations, Pandera, TFDV, Evidently

### Train/serve skew

One of the leading causes of production failures: features computed differently in training vs serving.

**Mitigations:**
1. Shared feature code (one library used by both paths)
2. Feature store (single source of truth)
3. Log serving features and replay them in training

### Pipelines

A pipeline is a DAG of steps: ingest → validate → feature → train → evaluate → register.

**Tools:** Airflow, Prefect, Dagster, Kubeflow Pipelines, Metaflow, ZenML

**Starting point:** If you can't run your training with one command (`make train` or `python -m src.pipeline.train`), you don't have a pipeline.

---

## 6. Packaging & Serving Models

### The packaging problem

A trained model in memory is useless in production. You need a **portable artifact** another process can load and call.

| Format | Notes |
|--------|-------|
| `pickle` / `joblib` | Easy; security risk with untrusted files; tied to library versions |
| **ONNX** | Cross-framework, runtime-optimized inference |
| **TorchScript** / SavedModel | Framework-native, production-grade |
| **MLflow Model** | Wraps any of the above with metadata + signature |

### Model signature

A model should declare what it accepts and returns — catches type errors before they crash silently:

```yaml
inputs:
  - {name: age,    type: long}
  - {name: income, type: double}
outputs:
  - {name: prob,   type: double}
```

### Serving patterns

| Pattern | Use case |
|---------|----------|
| **Batch** (offline, write to DB) | Daily recommendations, monthly risk scoring |
| **Online REST/gRPC** | Latency-sensitive: fraud, search, ads |
| **Streaming** | Continuous event scoring (Kafka, Flink) |
| **Embedded / on-device** | Mobile, IoT (ONNX, TFLite, CoreML) |

### Why FastAPI in MLOps?

**FastAPI** is a modern Python web framework used to wrap a trained model behind an HTTP API so other systems can request predictions over the network. In MLOps it serves as the "last mile" — the bridge between a model sitting in a file and a production application that needs real-time answers. You send a JSON request with input features, and the API returns the model's prediction. FastAPI is popular because it's fast, auto-generates API docs, and uses Pydantic for input validation (catching bad requests before they reach the model).

### Minimal FastAPI service

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.joblib")

class Features(BaseModel):
    mean_radius: float
    mean_texture: float

@app.post("/predict")
def predict(f: Features):
    proba = model.predict_proba([[f.mean_radius, f.mean_texture]])[0, 1]
    return {"probability": float(proba)}

@app.get("/health")
def health():
    return {"status": "ok"}
```

### Essential endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness — am I alive? |
| `GET /ready` | Readiness — model loaded, deps reachable? |
| `POST /predict` | Inference |
| `GET /metrics` | Prometheus exposition format |

### Containerization

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Rule:** The same image runs in dev, staging, and prod. Configuration via environment variables, not code.

---

## 7. CI/CD for Machine Learning

### What's different from regular CI/CD?

ML CI/CD validates not just **code**, but also:
- **Data** (schema, distribution)
- **Model quality** (does the new model beat the current production model?)
- **Operational behaviour** (latency, payload size, memory)

### Three pipelines (CI / CT / CD)

| Pipeline | Trigger | Action |
|----------|---------|--------|
| **CI** (code) | Code change (push/PR) | Build + unit test + lint |
| **CT** (continuous training) | Data or code change | Retrain + evaluate + register if better |
| **CD** (deploy) | New registered model | Canary / staged rollout |

### Testing layers for ML

| Layer | Example |
|-------|---------|
| **Unit** | `featurize_age(35) == 1` |
| **Data** | "no nulls in `user_id`", "label distribution within 5% of last week" |
| **Model** | "AUC on holdout ≥ 0.85", "AUC on protected group ≥ overall − 5%" |
| **Behavioural** | "increasing income increases predicted credit limit" (directional expectation) |
| **Integration** | "POST /predict returns 200 with valid payload in <100 ms" |
| **Smoke** | Hit `/health` after deploy |

### Safe deployment strategies

| Strategy | Description |
|----------|-------------|
| **Shadow / dark launch** | New model receives traffic, predictions not used; compare to prod |
| **Canary** | 1% → 10% → 100% with automated rollback on metric regression |
| **A/B test** | Randomized split, business metric is source of truth |
| **Blue / green** | Two environments, switch the router |

**Rule:** Never replace a model atomically in production without a rollback plan.

---

## 8. Monitoring & Drift Detection

### Why models decay

A trained model assumes the world keeps looking like the training set. It rarely does.

| Type | What changes | Example |
|------|--------------|---------|
| **Data drift** (covariate shift) | $P(X)$ | Users skew younger after a marketing campaign |
| **Concept drift** | $P(y \mid X)$ | Fraud patterns change after a new payment method |
| **Label drift** | $P(y)$ | Spam ratio drops from 30% to 5% |
| **Pipeline bugs** | Nothing "real" | An ETL job started filling nulls with 0 |

### Four layers of monitoring

1. **Service health** — uptime, latency, error rate (standard SRE)
2. **Data health** — input schema, null rates, feature distributions vs reference
3. **Model health** — prediction distribution, confidence, segment-level metrics
4. **Business KPIs** — conversion, revenue, complaints (the ultimate truth)

### Drift detection methods

**Numeric features:**
- **PSI (Population Stability Index)** — bin data, compare proportions
  - PSI < 0.1: stable
  - PSI 0.1–0.25: warning
  - PSI > 0.25: alert
- **Kolmogorov–Smirnov test** — non-parametric distribution test
- **Wasserstein distance** — distance between distributions

**Categorical features:**
- Chi-squared test
- JS divergence

### The label delay problem

You usually can't compute model quality (AUC, etc.) in real time because labels arrive late.

**Workarounds:**
- Proxy metrics: prediction distribution shifts
- Delayed evaluation: compute weekly with reconciled labels
- Active labelling: route a small random % for human labelling

### Alerting best practices

- Alert on **business symptoms**, not just statistical p-values
- Every alert must have a **runbook**: how to diagnose, who owns it, how to roll back
- Tune thresholds against **historical data** to avoid alert fatigue

---

## 9. Governance, Security & Team Structure

### Governance requirements (regulated domains)

For finance, healthcare, EU AI Act compliance, you must answer:

| Question | Practice |
|----------|----------|
| What data trained this model? With what consent? | Model cards, datasheets, lineage |
| Who approved the deployment? | Approval workflow in model registry |
| How does the model behave on protected groups? | Bias/fairness evaluation in CI |
| How quickly can you turn it off? | Rollback plan, kill switch |

### Security checklist

- **Don't** `pickle.load` untrusted artifacts — prefer signed artifacts or ONNX
- Treat the model as a **confidentiality risk** (membership inference, data extraction)
- **Validate and rate-limit** prediction inputs (adversarial examples, abuse)
- Secrets via environment variables / secret managers — **never** in notebooks or git
- Pin dependencies; scan for CVEs (`pip-audit`, Dependabot)

### Team structures

| Structure | Pros | Cons |
|-----------|------|------|
| **Embedded** (DS inside product team) | Fast iteration | Inconsistent practices |
| **Central platform team** | Consistent, reusable | Can become a bottleneck |
| **Hybrid** (central platform + embedded MLEs) | Balance of speed and consistency | More coordination needed |

---

## 10. The Non-Negotiables

These six principles are the minimum for any production ML system:

| # | Principle | What it means |
|---|-----------|---------------|
| 1 | **Reproducibility** | Code + data + environment are versioned together |
| 2 | **Automation** | One command retrains, one command deploys |
| 3 | **Testing** | Code, data, and model quality all gate releases |
| 4 | **Observability** | You know when the model is misbehaving before users do |
| 5 | **Reversibility** | Every deployment can be rolled back quickly |
| 6 | **Ownership** | Every model has a named owner and a runbook |

---

## 11. Key Tools Reference

| Concern | Open Source | Managed/Commercial |
|---------|-------------|-------------------|
| Experiment tracking | MLflow, W&B (free tier) | W&B, Neptune, Comet |
| Pipelines | Airflow, Prefect, Dagster, Kubeflow, Metaflow | Vertex AI Pipelines, SageMaker Pipelines |
| Feature store | Feast | Tecton, Vertex FS, SageMaker FS |
| Serving | BentoML, KServe, Triton, FastAPI | SageMaker Endpoints, Vertex Endpoints |
| Monitoring | Evidently, NannyML | Arize, WhyLabs, Fiddler |
| Data versioning | DVC, LakeFS | Delta Lake, Snowflake time travel |
| Data validation | Great Expectations, Pandera, TFDV | — |
| End-to-end | ZenML, Metaflow, MLflow | SageMaker, Vertex AI, Databricks |

**Rule:** Start with the smallest stack that solves your real problems. Adopt new tools only when current pain justifies it.

---

## 12. Further Reading

### Books
- Chip Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022) — the best single book on production ML
- Martin Kleppmann, *Designing Data-Intensive Applications* — data infrastructure fundamentals

### Papers
- D. Sculley et al., *Hidden Technical Debt in Machine Learning Systems* (NeurIPS 2015) — the foundational "ML systems" paper
- Google Cloud, *MLOps: Continuous delivery and automation pipelines in machine learning*

### Communities & Resources
- [Made With ML](https://madewithml.com/) — structured MLOps curriculum
- [MLOps Community](https://mlops.community/) — Slack, meetups, talks
- [Evidently AI Blog](https://evidentlyai.com/blog) — monitoring and drift deep-dives

---

## Study Checklist

Use this to self-assess your readiness:

- [ ] Can you explain MLOps maturity levels 0/1/2 and when each is appropriate?
- [ ] Can you list the four pillars of reproducibility and give a tool for each?
- [ ] Can you describe what MLflow's four components do?
- [ ] Can you explain train/serve skew and name two mitigations?
- [ ] Can you list three serving patterns and when to use each?
- [ ] Can you explain what CI/CT/CD mean in the ML context?
- [ ] Can you name the testing layers for ML (unit, data, model, behavioural, integration)?
- [ ] Can you define data drift vs concept drift vs label drift?
- [ ] Can you explain PSI and what thresholds indicate trouble?
- [ ] Can you describe three safe deployment strategies?
- [ ] Can you list the six non-negotiables for production ML?
