import json

import joblib

from src import bootstrap
from tests.conftest import make_raw

SPEC = {"name": "random_forest", "strategy": "class_weight",
        "params": {"clf__n_estimators": 20, "clf__max_depth": 4}}


def test_rebuilds_model_from_metadata(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    processed.mkdir()
    make_raw(300).to_csv(processed / "train.csv", index=False)
    monkeypatch.setattr(bootstrap, "PROCESSED_DIR", processed)
    (tmp_path / "metadata.json").write_text(json.dumps({"pipeline": SPEC}))

    assert bootstrap.ensure_model(tmp_path) is True
    model = joblib.load(tmp_path / "model.joblib")
    assert model.named_steps["clf"].n_estimators == 20
    assert model.named_steps["clf"].max_depth == 4
    assert bootstrap.ensure_model(tmp_path) is False  # already there: no rebuild
