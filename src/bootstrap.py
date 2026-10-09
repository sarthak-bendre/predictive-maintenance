"""Rebuild the production model from models/metadata.json when model.joblib is
missing (e.g. a fresh clone or a cloud-hosted dashboard).

metadata.json is committed and records the champion's pipeline (model type,
imbalance strategy, hyperparameters) and its decision threshold. All random
seeds are fixed, so refitting on the same training split reproduces the
champion exactly, without MLflow or a hyperparameter search.
"""
import json

import joblib
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR
from src.features import build_xy
from src.ingest import download
from src.modeling import make_model, prepare_split


def ensure_model(models_dir=MODELS_DIR) -> bool:
    """Make sure models_dir/model.joblib exists. Returns True if it was rebuilt."""
    if (models_dir / "model.joblib").exists():
        return False
    spec = json.loads((models_dir / "metadata.json").read_text())["pipeline"]

    if not (PROCESSED_DIR / "train.csv").exists():
        download()
        prepare_split()

    X, y = build_xy(pd.read_csv(PROCESSED_DIR / "train.csv"))
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    model = make_model(spec["name"], spec["strategy"], pos_weight).set_params(**spec["params"])
    model.fit(X, y)
    joblib.dump(model, models_dir / "model.joblib")
    return True


if __name__ == "__main__":
    print("rebuilt" if ensure_model() else "model already present")
