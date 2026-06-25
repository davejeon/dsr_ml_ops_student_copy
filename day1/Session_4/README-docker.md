# Running the iris_pipeline in Docker (all operating systems)

This is the recommended way to run the Session 4 Airflow demo. It gives every
student the **same result regardless of OS** (macOS, Linux, Windows) and avoids
the local-install problems some macOS versions have with the Airflow webserver.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and **running**
  (on Windows, use the WSL2 backend).

## Start it

From this directory (`day1/Session_4`):

```bash
docker compose up --build
```

The first run builds the image and downloads Postgres (a few minutes). When you
see `Listening at: http://0.0.0.0:8080`, open:

> http://localhost:8080  —  login **admin** / **admin**

Then trigger the **`iris_pipeline`** DAG (toggle it on, click ▶). Watch the
tasks go green: `load_data → validate_data → featurize → train_model → evaluate`.
The accuracy and classification report are printed in the `evaluate` task logs.

## Run the pipeline without the UI

Same container, one-shot CLI run (no clicking required):

```bash
docker compose run --rm airflow airflow dags test iris_pipeline
```

## Stop it

```bash
docker compose down        # stop containers, keep the database
docker compose down -v     # stop and wipe the database + logs (fresh start)
```

## How it works / what to edit

- **`Dockerfile`** – Airflow 2.9.2 on Python 3.12 (matches the course venv) plus
  `pandas`, `scikit-learn`, `numpy` from `requirements.txt`.
- **`docker-compose.yaml`** – two containers: `postgres` (metadata DB) and
  `airflow` (runs `db migrate`, the scheduler, and the webserver).
- The DAG file in `airflow_home/dags/` is **mounted live** — edit
  `iris_pipeline.py` on your machine and Airflow picks up the change (no rebuild).
- `requirements.txt` change → re-run `docker compose up --build`.

The dataset (`day1/Session_1/iris.csv`) is mounted at the exact path the DAG
computes from `__file__`, so the committed DAG code runs unmodified.
