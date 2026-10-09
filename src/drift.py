"""Data drift check with Evidently: training data vs. a batch of "new" data.

Two batches are compared against the training set:
- baseline: the held-out test set as-is (same distribution, should not drift)
- summer:   the test set with a simulated heatwave. Ambient air is +4 K, but
            the process only warms by +2.5 K because cooling works harder, so
            the process/air gap also shrinks, which is the heat-dissipation risk.
The script writes an HTML report per batch and prints which features drifted.
"""
import json
import sys

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import PROCESSED_DIR, REPORTS_DIR, SENSOR_COLS
from src.features import ENGINEERED_COLS, add_features

MONITORED = SENSOR_COLS + ENGINEERED_COLS
# Retrain if more than this share of monitored features drift.
DRIFT_SHARE_ALERT = 0.3


def simulate_summer(df: pd.DataFrame, air_shift: float = 4.0, process_shift: float = 2.5) -> pd.DataFrame:
    out = df.copy()
    out["air_temp_k"] += air_shift
    out["process_temp_k"] += process_shift
    return out


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, name: str) -> dict:
    ref = add_features(reference)[MONITORED]
    cur = add_features(current)[MONITORED]
    report = Report(metrics=[DataDriftPreset(drift_share=DRIFT_SHARE_ALERT)])
    report.run(reference_data=ref, current_data=cur)
    REPORTS_DIR.mkdir(exist_ok=True)
    report.save_html(str(REPORTS_DIR / f"drift_{name}.html"))

    table = next(m["result"] for m in report.as_dict()["metrics"]
                 if m["metric"] == "DataDriftTable")
    drifted = sorted(c for c, r in table["drift_by_columns"].items() if r["drift_detected"])
    share = table["share_of_drifted_columns"]
    return {
        "batch": name,
        "drifted_features": drifted,
        "share_drifted": round(share, 3),
        "retrain_recommended": share > DRIFT_SHARE_ALERT,
    }


def main() -> int:
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    results = [
        drift_report(train, test, "baseline"),
        drift_report(train, simulate_summer(test), "summer"),
    ]
    (REPORTS_DIR / "drift_summary.json").write_text(json.dumps(results, indent=2))
    for r in results:
        flag = "RETRAIN" if r["retrain_recommended"] else "ok"
        print(f"[{flag}] {r['batch']}: {r['share_drifted']:.0%} drifted -> {r['drifted_features']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
