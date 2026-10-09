"""Physics-based feature engineering, shared by training and the API.

The AI4I failure modes are defined by physical rules, so the features mirror them:
- temp_diff_k: heat-dissipation failure happens when the process/air gap is small
  and the spindle is slow.
- power_w: power failure happens when mechanical power (torque x angular speed)
  is outside a safe band.
- wear_torque: overstrain failure happens when tool wear x torque is high.
"""
import numpy as np
import pandas as pd

from src.config import ID_COLS, LEAKY_COLS, PRODUCT_TYPES, SENSOR_COLS, TARGET

ENGINEERED_COLS = ["temp_diff_k", "power_w", "wear_torque"]
TYPE_COLS = [f"type_{t}" for t in PRODUCT_TYPES]
FEATURE_COLS = SENSOR_COLS + ENGINEERED_COLS + TYPE_COLS


def drop_leaky(df: pd.DataFrame) -> pd.DataFrame:
    """Remove ID columns and the failure-mode flags that encode the label."""
    return df.drop(columns=[c for c in ID_COLS + LEAKY_COLS if c in df.columns])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    omega = out["rpm"] * 2 * np.pi / 60  # rpm -> rad/s
    out["power_w"] = out["torque_nm"] * omega
    out["wear_torque"] = out["tool_wear_min"] * out["torque_nm"]
    for t in PRODUCT_TYPES:
        out[f"type_{t}"] = (out["type"] == t).astype(int)
    return out


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = add_features(drop_leaky(df))
    return df[FEATURE_COLS], df[TARGET].astype(int)


def build_x(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for inference (no target needed)."""
    return add_features(df)[FEATURE_COLS]
