# airflow_home

This is the `AIRFLOW_HOME` for the Session 4 Airflow demo. The only file that
matters here is the DAG:

```
dags/iris_pipeline.py    # load_data → validate_data → featurize → train_model → evaluate
```

Runtime files (`airflow.cfg`, `airflow.db`, `logs/`, `standalone_admin_password.txt`,
`webserver_config.py`) are **not** committed — Airflow regenerates them, and they
are machine-specific.

## How to run

Use the Docker setup in the parent directory (`day1/Session_4`) — it works the
same on macOS, Linux and Windows. See **`../README-docker.md`**:

```bash
cd ..
docker compose up --build
# http://localhost:8080  (admin / admin) → trigger the "iris_pipeline" DAG
```

The DAG resolves `iris.csv` from `day1/Session_1/` via a path relative to its own
location, so it runs unmodified inside the container.
