import numpy as np
import pytest

from src.config import ID_COLS, LEAKY_COLS
from src.features import FEATURE_COLS, add_features, build_x, build_xy


def test_build_xy_drops_leaky_and_id_columns(raw_df):
    X, y = build_xy(raw_df)
    assert list(X.columns) == FEATURE_COLS
    assert not set(LEAKY_COLS + ID_COLS) & set(X.columns)
    assert len(X) == len(y) == len(raw_df)


def test_physics_features():
    import pandas as pd
    row = pd.DataFrame([{"type": "M", "air_temp_k": 300.0, "process_temp_k": 310.0,
                         "rpm": 60 / (2 * np.pi), "torque_nm": 50.0, "tool_wear_min": 100.0}])
    out = add_features(row).iloc[0]
    assert out.temp_diff_k == pytest.approx(10.0)
    assert out.power_w == pytest.approx(50.0)  # 1 rad/s * 50 Nm
    assert out.wear_torque == pytest.approx(5000.0)
    assert (out.type_L, out.type_M, out.type_H) == (0, 1, 0)


def test_build_x_works_without_target(raw_df):
    X = build_x(raw_df.drop(columns=["machine_failure"]))
    assert list(X.columns) == FEATURE_COLS
