FROM python:3.10-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# libgomp is needed by xgboost / scikit-learn at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY app/ app/
COPY models/model.joblib models/metadata.json models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
