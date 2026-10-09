import pytest

from src.drift import drift_report, simulate_summer


def test_summer_shift_shrinks_temperature_gap(raw_df):
    hot = simulate_summer(raw_df, air_shift=4.0, process_shift=2.5)
    gap_before = (raw_df.process_temp_k - raw_df.air_temp_k).mean()
    gap_after = (hot.process_temp_k - hot.air_temp_k).mean()
    assert gap_after == pytest.approx(gap_before - 1.5)


def test_drift_detected_only_for_shifted_batch(raw_df, tmp_path, monkeypatch):
    monkeypatch.setattr("src.drift.REPORTS_DIR", tmp_path)
    same = drift_report(raw_df, raw_df.sample(frac=1, random_state=1), "same")
    hot = drift_report(raw_df, simulate_summer(raw_df), "hot")
    assert same["drifted_features"] == []
    assert "air_temp_k" in hot["drifted_features"]
