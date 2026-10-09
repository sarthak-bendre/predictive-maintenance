"""FastAPI service: sensor readings in, failure probability and decision out."""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.features import build_x

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = Path(os.getenv("MODELS_DIR", DEFAULT_MODELS_DIR))
    model_path = models_dir / "model.joblib"
    if model_path.exists():
        state["model"] = joblib.load(model_path)
        state["meta"] = json.loads((models_dir / "metadata.json").read_text())
    yield
    state.clear()


app = FastAPI(title="Predictive Maintenance API", version="1.0.0", lifespan=lifespan)


class SensorReading(BaseModel):
    """One machine snapshot. Bounds match src/validate.py."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "type": "L",
                "air_temp_k": 298.1,
                "process_temp_k": 308.6,
                "rpm": 1551,
                "torque_nm": 42.8,
                "tool_wear_min": 108,
            }
        },
    )

    type: Literal["L", "M", "H"]
    air_temp_k: float = Field(ge=250, le=350)
    process_temp_k: float = Field(ge=250, le=400)
    rpm: float = Field(ge=0, le=5000)
    torque_nm: float = Field(ge=0, le=150)
    tool_wear_min: float = Field(ge=0, le=400)

    @model_validator(mode="after")
    def process_hotter_than_air(self):
        if self.process_temp_k < self.air_temp_k:
            raise ValueError("process_temp_k must be >= air_temp_k")
        return self


class Prediction(BaseModel):
    failure_probability: float
    threshold: float
    failure_predicted: bool
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=Prediction)
def predict(reading: SensorReading):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    meta = state["meta"]
    X = build_x(pd.DataFrame([reading.model_dump()]))
    proba = float(state["model"].predict_proba(X)[0, 1])
    return Prediction(
        failure_probability=round(proba, 4),
        threshold=meta["threshold"],
        failure_predicted=proba >= meta["threshold"],
        model_version=f"{meta['registered_model']}/v{meta['registered_version']} ({meta['model']})",
    )
