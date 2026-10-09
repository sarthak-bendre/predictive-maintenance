PY := .venv/bin/python
export MLFLOW_DISABLE_AGENT_HINT=1

.PHONY: setup pipeline data train promote explain drift test serve mlflow docker

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

pipeline: data train promote explain drift

data:
	$(PY) -m src.ingest
	$(PY) -m src.validate

train:
	$(PY) -m src.train

promote:
	$(PY) -m src.promote

explain:
	$(PY) -m src.explain

drift:
	$(PY) -m src.drift

test:
	$(PY) -m pytest -v

serve:
	.venv/bin/uvicorn app.main:app --reload --port 8000

mlflow:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker:
	docker build -t predictive-maintenance .
	docker run --rm -p 8000:8000 predictive-maintenance
