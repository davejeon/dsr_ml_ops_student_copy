# Day 2 — Session 3: CI/CD for the Iris Classifier

Hands-on CI/CD for the iris service built in Sessions 1 and 2.

**Prerequisites:**
- `day2/Session_1/train.py` has been run (produces `.joblib` files)
- `.venv` activated (`source .venv/bin/activate` from repo root)

---

### Start

| File | What it does |
|------|-------------|
| `conftest.py` | Shared fixtures and constants used by all test files |
| `test_data.py` | 4 tests: columns present, no nulls, known labels, row count |
| `test_model.py` | 3 tests: accuracy threshold, two canonical predictions |
| `test_service.py` | 3 tests: health check, correct prediction, bad input rejected |
| `ml-ci.yml` | Single-job GitHub Actions workflow: install → train → test |

---

## Running the tests

All commands from the **repo root** with the venv activated.

### Data + model tests (no service needed)

```bash
pytest day2/Session_3/test_data.py day2/Session_3/test_model.py -v
```

### Integration tests (service must be running)

```bash
# Terminal 1
uvicorn service:app --app-dir day2/Session_1 --port 8000

# Terminal 2
pytest day2/Session_3/test_service.py -v
```

If the service is not running, integration tests are **skipped** (not failed).

### Everything at once

```bash
# Start the service first, then:
pytest day2/Session_3/ -v
```

---

## Deliberately breaking a test

The best way to understand what a test guards against is to make it fail:

```bash
# 1. Open iris.csv and change one value to -1.0
#    → test_data.py::test_no_missing_values or a range test in the advanced file will catch it

# 2. Raise ACCURACY_THRESHOLD in test_model.py to 0.99
#    → test_model.py::test_accuracy_above_threshold will fail

# 3. Send a bad request to the running service:
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"septal_length": -1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
# → 422 Unprocessable Entity  (test_service.py::test_invalid_input_is_rejected_with_422 passes)
```

---

## Using the GitHub Actions workflow

```bash
mkdir -p .github/workflows
cp day2/Session_3/ml-ci.yml .github/workflows/ml-ci.yml
git add .github/workflows/ml-ci.yml
git commit -m "add ML CI pipeline"
git push
```

The basic workflow runs one job on every push and PR: install → train → data tests → model tests.
