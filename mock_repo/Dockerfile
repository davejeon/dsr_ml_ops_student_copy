FROM python:3.11-slim

WORKDIR /app

# install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source + trained model
COPY src/ ./src/
COPY model.joblib ./model.joblib

ENV PYTHONPATH=/app/src
EXPOSE 8000

# same image runs in dev, staging, prod — config via env vars
CMD ["uvicorn", "churn.inference.service:app", "--host", "0.0.0.0", "--port", "8000"]
