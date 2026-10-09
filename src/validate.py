"""Data validation: schema, types and physically sensible ranges.

Plain checks rather than a framework, so every rule is visible in one place.
The ranges are generous physical bounds, not the min/max of the training data,
so genuinely new conditions (e.g. a hot summer) pass and are left to drift
monitoring, while impossible values (negative torque) are rejected.
"""
import pandas as pd

from src.config import PRODUCT_TYPES, TARGET

# column -> (min, max), inclusive
RANGES = {
    "air_temp_k": (250.0, 350.0),
    "process_temp_k": (250.0, 400.0),
    "rpm": (0.0, 5000.0),
    "torque_nm": (0.0, 150.0),
    "tool_wear_min": (0.0, 400.0),
}


class DataValidationError(ValueError):
    pass


def validate(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    errors = []

    required = list(RANGES) + ["type"] + ([TARGET] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")

    if df.empty:
        raise DataValidationError("Dataset is empty")

    nulls = df[required].isna().sum()
    if nulls.any():
        errors.append(f"Null values: {nulls[nulls > 0].to_dict()}")

    for col, (lo, hi) in RANGES.items():
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"{col} is not numeric (dtype={df[col].dtype})")
            continue
        bad = (~df[col].between(lo, hi)).sum()
        if bad:
            errors.append(f"{col}: {bad} rows outside [{lo}, {hi}]")

    bad_types = set(df["type"].dropna().unique()) - set(PRODUCT_TYPES)
    if bad_types:
        errors.append(f"Unknown product types: {sorted(bad_types)}")

    if require_target:
        bad_target = set(df[TARGET].dropna().unique()) - {0, 1}
        if bad_target:
            errors.append(f"Target has non-binary values: {sorted(bad_target)}")

    if errors:
        raise DataValidationError("; ".join(errors))
    return df


if __name__ == "__main__":
    from src.ingest import load_raw

    validate(load_raw())
    print("Validation passed")
