# Mock Repo — `churn-prediction`

A **reference scaffold** that mirrors the project structure described in
`mlops_learning_manual.md` (Section 3). Nothing here needs to run — it's a
visual example of how the pieces fit together in a small production-shaped ML
project.

> Domain: predicting customer churn (same example used in the manual & Day 1).

## Layout

```
mock_repo/
├── README.md                  # you are here
├── pyproject.toml             # package metadata + pinned deps
├── requirements.txt           # flat pinned deps (CI/Docker)
├── Dockerfile                 # container for the serving API
├── Makefile                   # one-command train / serve / test
├── dvc.yaml                   # data + pipeline versioning
├── .github/
│   └── workflows/
│       └── ci.yml             # CI: lint + unit + data + model tests
├── data/                      # gitignored in real life
│   ├── raw/.gitkeep
│   ├── interim/.gitkeep
│   └── processed/.gitkeep
├── notebooks/
│   └── 01_exploration.ipynb   # exploration only — no production code
├── src/
│   └── churn/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── load.py
│       │   └── validate.py    # schema + distribution checks
│       ├── features/
│       │   ├── __init__.py
│       │   └── build.py       # shared by train + serve (no skew)
│       ├── models/
│       │   ├── __init__.py
│       │   └── train.py       # MLflow-tracked training entrypoint
│       └── inference/
│           ├── __init__.py
│           └── service.py     # FastAPI app
└── tests/
    ├── test_features.py       # unit
    ├── test_data.py           # data-layer
    ├── test_model.py          # model-quality gate
    └── test_service.py        # integration (TestClient)
```

## How the pieces map to the manual

| Manual section | Files in this repo |
|---|---|
| 3. Reproducibility | `pyproject.toml`, `requirements.txt`, `Dockerfile`, seed in `train.py` |
| 4. Experiment tracking | `src/churn/models/train.py` (MLflow logging) |
| 5. Data versioning & validation | `dvc.yaml`, `src/churn/data/validate.py`, `tests/test_data.py` |
| 6. Packaging & serving | `src/churn/inference/service.py`, `Dockerfile` |
| 7. CI/CD | `.github/workflows/ci.yml`, `tests/test_model.py` |
| 8. Monitoring | `/metrics` endpoint stub in `service.py` |

## Quick tour

- `make train` → trains a baseline, logs to MLflow, writes `model.joblib`.
- `make serve` → starts FastAPI on `:8000` with `/health`, `/predict`, `/metrics`.
- `make test`  → runs all four test layers (unit, data, model, integration).
