import numpy as np
import pytest

from src.validate import DataValidationError, validate


def test_valid_data_passes(raw_df):
    validate(raw_df)


@pytest.mark.parametrize("col,value", [
    ("torque_nm", -1.0),
    ("rpm", 99999.0),
    ("tool_wear_min", -5.0),
    ("air_temp_k", np.nan),
])
def test_out_of_range_or_null_rejected(raw_df, col, value):
    raw_df.loc[0, col] = value
    with pytest.raises(DataValidationError):
        validate(raw_df)


def test_missing_column_rejected(raw_df):
    with pytest.raises(DataValidationError, match="Missing columns"):
        validate(raw_df.drop(columns=["torque_nm"]))


def test_unknown_product_type_rejected(raw_df):
    raw_df.loc[0, "type"] = "X"
    with pytest.raises(DataValidationError, match="product types"):
        validate(raw_df)


def test_target_optional_for_inference(raw_df):
    validate(raw_df.drop(columns=["machine_failure"]), require_target=False)
