import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLS, build_xy


def make_raw(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic rows in the cleaned raw schema (no real data needed in CI)."""
    rng = np.random.default_rng(seed)
    air = rng.normal(300, 2, n)
    df = pd.DataFrame({
        "udi": np.arange(n),
        "product_id": [f"L{i}" for i in range(n)],
        "type": rng.choice(["L", "M", "H"], n),
        "air_temp_k": air,
        "process_temp_k": air + rng.normal(10, 1, n),
        "rpm": rng.normal(1500, 150, n),
        "torque_nm": rng.normal(40, 10, n).clip(1, 100),
        "tool_wear_min": rng.uniform(0, 250, n),
    })
    df["machine_failure"] = ((df.tool_wear_min * df.torque_nm) > 9000).astype(int)
    for c in ["twf", "hdf", "pwf", "osf", "rnf"]:
        df[c] = 0
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """A small fitted model + metadata, wired into the API via MODELS_DIR."""
    X, y = build_xy(make_raw(500))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text(json.dumps({
        "model": "test", "threshold": 0.5, "features": FEATURE_COLS,
        "registered_model": "pm-failure-classifier", "registered_version": 0,
    }))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    return tmp_path
