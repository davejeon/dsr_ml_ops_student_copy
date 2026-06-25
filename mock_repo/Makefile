.PHONY: install train serve test lint docker

install:
	pip install -e ".[dev]"

train:
	python -m churn.models.train

serve:
	uvicorn churn.inference.service:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest --cov=churn

lint:
	ruff check src tests

docker:
	docker build -t churn-svc:0.1.0 .
