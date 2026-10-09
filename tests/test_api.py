import pytest
from fastapi.testclient import TestClient

from app.main import app

VALID = {"type": "L", "air_temp_k": 298.1, "process_temp_k": 308.6,
         "rpm": 1551, "torque_nm": 42.8, "tool_wear_min": 108}


@pytest.fixture
def client(models_dir):
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_valid_response(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["failure_predicted"] == (body["failure_probability"] >= body["threshold"])


@pytest.mark.parametrize("bad", [
    {**VALID, "torque_nm": -1},
    {**VALID, "type": "X"},
    {**VALID, "process_temp_k": 290.0},          # colder than air
    {**VALID, "extra_field": 1},
    {k: v for k, v in VALID.items() if k != "rpm"},
])
def test_bad_input_rejected(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_503_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/predict", json=VALID).status_code == 503
