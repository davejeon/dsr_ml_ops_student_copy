# Minimal Airflow image for the Session 4 iris_pipeline demo.
# Same Airflow + Python as the course venv (2.9.2 / 3.12) so behaviour matches.
FROM apache/airflow:2.9.2-python3.12

# Add the ML libraries the DAG imports (pandas / scikit-learn / numpy).
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
